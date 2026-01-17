#include "phantomcore/spike_detector.hpp"
#include "phantomcore/bandpass_filter.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace phantomcore {

// ============================================================================
// SpikeDetector Implementation
// ============================================================================

struct SpikeDetector::Impl {
    size_t num_channels = 0;
    
    // Per-channel state (dynamic allocation)
    std::vector<float> thresholds;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    std::vector<size_t> refractory_counter;
    std::vector<bool> channel_enabled;
    
    // DSP: Bandpass filter bank (one filter per channel)
    std::unique_ptr<DynamicBandpassFilterBank> filter_bank;
    bool filtering_enabled = true;
    
    // Statistics
    Stats stats;
    LatencyTracker latency_tracker{1000};
    size_t sample_count = 0;
    
    explicit Impl(size_t channels) : num_channels(channels) {
        thresholds.resize(channels, 0.0f);
        running_mean.resize(channels, 0.0f);
        running_var.resize(channels, 1.0f);
        refractory_counter.resize(channels, 0);
        channel_enabled.resize(channels, true);
        stats.spikes_per_channel.resize(channels, 0);
    }
    
    void update_adaptive_threshold(size_t channel, float value, double rate) {
        // Welford's online algorithm for FILTERED signal statistics
        sample_count++;
        float delta = value - running_mean[channel];
        running_mean[channel] += delta / static_cast<float>(sample_count);
        float delta2 = value - running_mean[channel];
        running_var[channel] += delta * delta2;
        
        // Exponential moving average for threshold
        float std = std::sqrt(running_var[channel] / 
                             std::max(1.0f, static_cast<float>(sample_count)));
        float target_threshold = running_mean[channel] + thresholds[channel] * std;
        thresholds[channel] += static_cast<float>(rate) * (target_threshold - thresholds[channel]);
    }
};

SpikeDetector::SpikeDetector(
    const ChannelConfig& channel_config,
    const SpikeDetectorConfig& config
)
    : impl_(std::make_unique<Impl>(channel_config.num_channels))
    , channel_config_(channel_config)
    , config_(config) 
{
    // Initialize thresholds based on config
    std::fill(impl_->thresholds.begin(), impl_->thresholds.end(), config.threshold_std);
    
    // Initialize bandpass filter bank
    impl_->filtering_enabled = config.use_bandpass_filter;
    if (impl_->filtering_enabled) {
        ButterworthBandpass::Config filter_cfg;
        filter_cfg.sample_rate = config.sample_rate;
        filter_cfg.low_cutoff = config.bandpass_low;
        filter_cfg.high_cutoff = config.bandpass_high;
        filter_cfg.order = config.filter_order;
        
        impl_->filter_bank = std::make_unique<DynamicBandpassFilterBank>(
            channel_config.num_channels, filter_cfg
        );
    }
}

// Legacy constructor for backward compatibility
SpikeDetector::SpikeDetector(const SpikeDetectorConfig& config)
    : SpikeDetector(ChannelConfig::mc_maze(), config) {}

SpikeDetector::~SpikeDetector() = default;
SpikeDetector::SpikeDetector(SpikeDetector&&) noexcept = default;
SpikeDetector& SpikeDetector::operator=(SpikeDetector&&) noexcept = default;

std::vector<SpikeEvent> SpikeDetector::process_sample(
    std::span<const float> sample,
    double timestamp
) {
    auto start = Clock::now();
    std::vector<SpikeEvent> events;
    
    const size_t num_channels = std::min(sample.size(), impl_->num_channels);
    
    for (size_t ch = 0; ch < num_channels; ++ch) {
        if (!impl_->channel_enabled[ch]) continue;
        
        // Decrement refractory counter
        if (impl_->refractory_counter[ch] > 0) {
            impl_->refractory_counter[ch]--;
            continue;
        }
        
        float value = sample[ch];
        
        // =====================================================================
        // CRITICAL DSP STEP: Bandpass Filter (300-3000 Hz)
        // =====================================================================
        // Raw neural signals contain:
        // - LFP oscillations (< 300 Hz) - slow waves, movement artifacts
        // - Action potentials (300-3000 Hz) - the actual spikes we want
        // - High-frequency noise (> 3000 Hz) - EMG, electrical interference
        //
        // Without filtering, threshold detection triggers on artifacts!
        // =====================================================================
        if (impl_->filtering_enabled && impl_->filter_bank) {
            value = impl_->filter_bank->process(ch, value);
        }
        
        // Adaptive threshold update (on FILTERED signal)
        if (config_.use_adaptive_threshold) {
            impl_->update_adaptive_threshold(ch, value, config_.adaptation_rate);
        }
        
        // Threshold crossing detection (negative threshold for spikes)
        float std = std::sqrt(impl_->running_var[ch] / 
                             std::max(1.0f, static_cast<float>(impl_->sample_count)));
        float threshold = impl_->running_mean[ch] + config_.threshold_std * std;
        
        if (value < threshold) {
            SpikeEvent event;
            event.channel = static_cast<uint32_t>(ch);
            event.timestamp = timestamp;
            event.amplitude = value;
            event.cluster_id = 0;  // Unsorted
            
            events.push_back(event);
            impl_->refractory_counter[ch] = config_.refractory_samples;
            impl_->stats.total_spikes_detected++;
            impl_->stats.spikes_per_channel[ch]++;
        }
    }
    
    impl_->latency_tracker.record(Clock::now() - start);
    return events;
}

std::vector<SpikeEvent> SpikeDetector::process_batch(
    std::span<const float> samples,
    size_t num_samples,
    size_t num_channels,
    double start_timestamp,
    double sample_rate
) {
    std::vector<SpikeEvent> all_events;
    double dt = 1.0 / sample_rate;
    
    for (size_t s = 0; s < num_samples; ++s) {
        double timestamp = start_timestamp + s * dt;
        std::span<const float> sample(
            samples.data() + s * num_channels,
            num_channels
        );
        
        auto events = process_sample(sample, timestamp);
        all_events.insert(all_events.end(), events.begin(), events.end());
    }
    
    return all_events;
}

std::vector<SpikeEvent> SpikeDetector::process_spike_counts(
    const SpikeCountArray& spike_counts,
    double timestamp
) {
    // For pre-binned spike counts, we simply convert to events
    // Each count > 0 indicates spike activity in that bin
    std::vector<SpikeEvent> events;
    
    // Increment sample count for stats tracking
    impl_->sample_count++;
    
    const size_t num_channels = std::min(spike_counts.size(), impl_->num_channels);
    for (size_t ch = 0; ch < num_channels; ++ch) {
        if (!impl_->channel_enabled[ch]) continue;
        
        int32_t count = spike_counts[ch];
        for (int32_t i = 0; i < count; ++i) {
            SpikeEvent event;
            event.channel = static_cast<uint32_t>(ch);
            event.timestamp = timestamp;
            event.amplitude = static_cast<float>(count);
            event.cluster_id = 0;
            
            events.push_back(event);
            impl_->stats.total_spikes_detected++;
            impl_->stats.spikes_per_channel[ch]++;
        }
    }
    
    return events;
}

void SpikeDetector::reset() {
    // Reset statistics and thresholds
    impl_ = std::make_unique<Impl>(channel_config_.num_channels);
    std::fill(impl_->thresholds.begin(), impl_->thresholds.end(), config_.threshold_std);
    
    // Reinitialize bandpass filter bank with fresh state
    impl_->filtering_enabled = config_.use_bandpass_filter;
    if (impl_->filtering_enabled) {
        ButterworthBandpass::Config filter_cfg;
        filter_cfg.sample_rate = config_.sample_rate;
        filter_cfg.low_cutoff = config_.bandpass_low;
        filter_cfg.high_cutoff = config_.bandpass_high;
        filter_cfg.order = config_.filter_order;
        
        impl_->filter_bank = std::make_unique<DynamicBandpassFilterBank>(
            channel_config_.num_channels, filter_cfg
        );
    }
}

std::span<const float> SpikeDetector::get_thresholds() const {
    return impl_->thresholds;
}

SpikeDetector::Stats SpikeDetector::get_stats() const {
    Stats stats = impl_->stats;
    
    // Calculate mean rate
    double total_time = impl_->sample_count / STREAM_RATE_HZ;
    if (total_time > 0) {
        stats.mean_rate_hz = static_cast<double>(stats.total_spikes_detected) / total_time;
    }
    
    // Get mean processing time
    auto latency_stats = impl_->latency_tracker.get_stats();
    stats.mean_processing_time = Duration(
        static_cast<int64_t>(latency_stats.mean_us * 1000)
    );
    
    return stats;
}

void SpikeDetector::set_channel_threshold(size_t channel, float threshold_std) {
    if (channel < impl_->num_channels) {
        impl_->thresholds[channel] = threshold_std;
    }
}

void SpikeDetector::set_channel_enabled(size_t channel, bool enabled) {
    if (channel < impl_->num_channels) {
        impl_->channel_enabled[channel] = enabled;
    }
}

// ============================================================================
// SpikeSorter Implementation
// ============================================================================

struct SpikeSorter::Impl {
    std::vector<Waveform> training_waveforms;
    std::vector<std::array<float, 48>> cluster_centroids;
    
    // PCA components (simplified - in production use SVD)
    std::array<std::array<float, 48>, 3> pca_components{};
    std::array<float, 48> pca_mean{};
};

SpikeSorter::SpikeSorter(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    impl_->training_waveforms.reserve(config.max_waveforms);
    impl_->cluster_centroids.resize(config.num_clusters);
}

SpikeSorter::SpikeSorter()
    : SpikeSorter(Config()) {}

SpikeSorter::~SpikeSorter() = default;

void SpikeSorter::add_waveform(const Waveform& waveform) {
    if (impl_->training_waveforms.size() < config_.max_waveforms) {
        impl_->training_waveforms.push_back(waveform);
    }
}

bool SpikeSorter::train() {
    if (impl_->training_waveforms.size() < config_.num_clusters * 10) {
        return false;  // Not enough data
    }
    
    // Compute mean waveform
    impl_->pca_mean.fill(0.0f);
    for (const auto& wf : impl_->training_waveforms) {
        for (size_t i = 0; i < 48; ++i) {
            impl_->pca_mean[i] += wf.samples[i];
        }
    }
    for (size_t i = 0; i < 48; ++i) {
        impl_->pca_mean[i] /= static_cast<float>(impl_->training_waveforms.size());
    }
    
    // Simple k-means clustering (without full PCA for simplicity)
    // In production, would use Eigen SVD for proper PCA
    
    // Initialize centroids with random waveforms
    for (size_t c = 0; c < config_.num_clusters; ++c) {
        size_t idx = (c * impl_->training_waveforms.size()) / config_.num_clusters;
        impl_->cluster_centroids[c] = impl_->training_waveforms[idx].samples;
    }
    
    // K-means iterations
    for (size_t iter = 0; iter < 100; ++iter) {
        // Assign waveforms to nearest centroid
        std::vector<std::vector<size_t>> assignments(config_.num_clusters);
        
        for (size_t w = 0; w < impl_->training_waveforms.size(); ++w) {
            float min_dist = std::numeric_limits<float>::max();
            size_t best_cluster = 0;
            
            for (size_t c = 0; c < config_.num_clusters; ++c) {
                float dist = 0.0f;
                for (size_t i = 0; i < 48; ++i) {
                    float diff = impl_->training_waveforms[w].samples[i] - 
                                impl_->cluster_centroids[c][i];
                    dist += diff * diff;
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = c;
                }
            }
            assignments[best_cluster].push_back(w);
        }
        
        // Update centroids
        bool converged = true;
        for (size_t c = 0; c < config_.num_clusters; ++c) {
            if (assignments[c].empty()) continue;
            
            std::array<float, 48> new_centroid{};
            for (size_t w : assignments[c]) {
                for (size_t i = 0; i < 48; ++i) {
                    new_centroid[i] += impl_->training_waveforms[w].samples[i];
                }
            }
            for (size_t i = 0; i < 48; ++i) {
                new_centroid[i] /= static_cast<float>(assignments[c].size());
                
                if (std::abs(new_centroid[i] - impl_->cluster_centroids[c][i]) > 
                    config_.convergence_threshold) {
                    converged = false;
                }
            }
            impl_->cluster_centroids[c] = new_centroid;
        }
        
        if (converged) break;
    }
    
    trained_ = true;
    return true;
}

uint32_t SpikeSorter::classify(const Waveform& waveform) const {
    if (!trained_) return 0;
    
    float min_dist = std::numeric_limits<float>::max();
    uint32_t best_cluster = 0;
    
    for (size_t c = 0; c < config_.num_clusters; ++c) {
        float dist = 0.0f;
        for (size_t i = 0; i < 48; ++i) {
            float diff = waveform.samples[i] - impl_->cluster_centroids[c][i];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = static_cast<uint32_t>(c);
        }
    }
    
    return best_cluster;
}

void SpikeSorter::reset() {
    impl_->training_waveforms.clear();
    trained_ = false;
}

}  // namespace phantomcore
