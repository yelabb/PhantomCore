#pragma once

#include "types.hpp"
#include "simd_utils.hpp"
#include <memory>
#include <span>
#include <vector>

namespace phantomcore {

/**
 * @brief High-performance spike detection engine with SIMD optimization
 * 
 * Implements threshold-based spike detection with:
 * - Adaptive thresholding
 * - Refractory period enforcement
 * - SIMD-accelerated processing
 * - Waveform extraction for sorting
 */
class SpikeDetector {
public:
    explicit SpikeDetector(const SpikeDetectorConfig& config = {});
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
        std::array<uint64_t, NUM_CHANNELS> spikes_per_channel{};
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
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
    };
    
    explicit SpikeSorter(const Config& config = {});
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
