#pragma once

#include "types.hpp"
#include "simd_utils.hpp"
#include "bandpass_filter.hpp"
#include <memory>
#include <span>
#include <vector>

namespace phantomcore {

/**
 * @brief High-performance spike detection engine with SIMD optimization
 * 
 * Implements a proper neural signal processing pipeline:
 * 
 *   Raw Signal → [Bandpass Filter] → [Threshold Detection] → Spike Events
 *                 300-3000 Hz          Adaptive threshold
 *                 Butterworth           with refractory
 * 
 * The bandpass filter is CRITICAL for BCI applications:
 * - Removes LFP (< 300 Hz): Slow oscillations, movement artifacts
 * - Removes high-freq noise (> 3000 Hz): EMG, electrical interference
 * - Isolates action potential frequency band
 * 
 * Supports runtime-configurable channel count for different hardware.
 */
class SpikeDetector {
public:
    /**
     * @brief Construct detector with channel configuration
     * @param channel_config Hardware-specific channel setup
     * @param config Detection parameters
     */
    explicit SpikeDetector(
        const ChannelConfig& channel_config = ChannelConfig::mc_maze(),
        const SpikeDetectorConfig& config = {}
    );
    
    /// Legacy constructor for backward compatibility
    explicit SpikeDetector(const SpikeDetectorConfig& config);
    
    ~SpikeDetector();
    
    // Non-copyable, movable
    SpikeDetector(const SpikeDetector&) = delete;
    SpikeDetector& operator=(const SpikeDetector&) = delete;
    SpikeDetector(SpikeDetector&&) noexcept;
    SpikeDetector& operator=(SpikeDetector&&) noexcept;
    
    /**
     * @brief Process a single multi-channel sample
     * @param sample Raw voltage sample for all channels
     * @param timestamp Sample timestamp
     * @return Vector of detected spike events (may be empty)
     */
    std::vector<SpikeEvent> process_sample(
        std::span<const float> sample,
        double timestamp
    );
    
    /**
     * @brief Process a batch of samples for efficiency
     * @param samples 2D buffer [num_samples x num_channels]
     * @param start_timestamp Timestamp of first sample
     * @param sample_rate Samples per second
     * @return All detected spikes in the batch
     */
    std::vector<SpikeEvent> process_batch(
        std::span<const float> samples,
        size_t num_samples,
        size_t num_channels,
        double start_timestamp,
        double sample_rate
    );
    
    /**
     * @brief Process spike counts from PhantomLink packets
     * This is a simplified interface for binned spike data
     * @param spike_counts Pre-binned spike counts per channel
     * @param timestamp Packet timestamp
     * @return Processed spike events
     */
    std::vector<SpikeEvent> process_spike_counts(
        const SpikeCountArray& spike_counts,
        double timestamp
    );
    
    /**
     * @brief Reset detector state
     */
    void reset();
    
    /**
     * @brief Get current per-channel thresholds
     */
    std::span<const float> get_thresholds() const;
    
    /**
     * @brief Get detection statistics
     */
    struct Stats {
        uint64_t total_spikes_detected = 0;
        std::vector<uint64_t> spikes_per_channel;  // Dynamic size
        double mean_rate_hz = 0.0;
        Duration mean_processing_time{};
    };
    Stats get_stats() const;
    
    /**
     * @brief Set channel-specific parameters
     */
    void set_channel_threshold(size_t channel, float threshold_std);
    void set_channel_enabled(size_t channel, bool enabled);
    
    /**
     * @brief Get configuration
     */
    const SpikeDetectorConfig& config() const { return config_; }
    
    /**
     * @brief Get channel configuration
     */
    const ChannelConfig& channel_config() const { return channel_config_; }
    
    /**
     * @brief Get number of channels
     */
    size_t num_channels() const { return channel_config_.num_channels; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ChannelConfig channel_config_;
    SpikeDetectorConfig config_;
};

/**
 * @brief Simple waveform-based spike sorter
 * Uses PCA + k-means for unsupervised clustering
 */
class SpikeSorter {
public:
    struct Config {
        size_t num_clusters = 3;          // Expected number of units
        size_t pca_components = 3;        // PCA dimensions
        size_t max_waveforms = 10000;     // Max waveforms for training
        double convergence_threshold = 0.001;
        
        Config() = default;
    };
    
    explicit SpikeSorter(const Config& config);
    SpikeSorter();
    ~SpikeSorter();
    
    /**
     * @brief Add waveform for training
     */
    void add_waveform(const Waveform& waveform);
    
    /**
     * @brief Train the sorter on collected waveforms
     * @return True if training converged
     */
    bool train();
    
    /**
     * @brief Classify a waveform to a cluster
     * @return Cluster ID (0 to num_clusters-1)
     */
    uint32_t classify(const Waveform& waveform) const;
    
    /**
     * @brief Check if sorter is trained
     */
    bool is_trained() const { return trained_; }
    
    /**
     * @brief Reset sorter state
     */
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    bool trained_ = false;
};

}  // namespace phantomcore
