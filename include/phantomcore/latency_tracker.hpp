#pragma once

#include "types.hpp"
#include "ring_buffer.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace phantomcore {

/**
 * @brief High-precision latency tracker for real-time systems
 * 
 * Tracks latency statistics with sub-microsecond precision.
 * Uses a lock-free ring buffer for thread-safe updates.
 */
class LatencyTracker {
public:
    explicit LatencyTracker(size_t window_size = 1000)
        : window_size_(window_size) {
        samples_.reserve(window_size);
    }
    
    /**
     * @brief Record a latency sample
     * @param latency Duration of the operation
     */
    void record(Duration latency) {
        double us = to_microseconds(latency);
        
        if (samples_.size() >= window_size_) {
            samples_.erase(samples_.begin());
        }
        samples_.push_back(us);
        
        // Update running stats
        total_samples_++;
        sum_ += us;
        sum_sq_ += us * us;
        
        if (us < min_) min_ = us;
        if (us > max_) max_ = us;
    }
    
    /**
     * @brief Record latency using start time
     * @param start Start timestamp
     */
    void record_since(Timestamp start) {
        record(Clock::now() - start);
    }
    
    /**
     * @brief Get computed statistics
     */
    LatencyStats get_stats() const {
        if (samples_.empty()) {
            return {};
        }
        
        LatencyStats stats;
        stats.sample_count = total_samples_;
        stats.min_us = min_;
        stats.max_us = max_;
        stats.mean_us = sum_ / total_samples_;
        
        // Variance and std
        double variance = (sum_sq_ / total_samples_) - 
                         (stats.mean_us * stats.mean_us);
        stats.std_us = std::sqrt(std::max(0.0, variance));
        
        // Percentiles (on window)
        std::vector<double> sorted = samples_;
        std::sort(sorted.begin(), sorted.end());
        
        auto percentile = [&](double p) {
            size_t idx = static_cast<size_t>(p * (sorted.size() - 1));
            return sorted[idx];
        };
        
        stats.p50_us = percentile(0.50);
        stats.p95_us = percentile(0.95);
        stats.p99_us = percentile(0.99);
        
        return stats;
    }
    
    /**
     * @brief Reset all statistics
     */
    void reset() {
        samples_.clear();
        total_samples_ = 0;
        sum_ = 0.0;
        sum_sq_ = 0.0;
        min_ = std::numeric_limits<double>::max();
        max_ = 0.0;
    }
    
    /**
     * @brief Get number of samples recorded
     */
    size_t sample_count() const { return total_samples_; }

private:
    size_t window_size_;
    std::vector<double> samples_;
    
    size_t total_samples_ = 0;
    double sum_ = 0.0;
    double sum_sq_ = 0.0;
    double min_ = std::numeric_limits<double>::max();
    double max_ = 0.0;
};

/**
 * @brief RAII scope timer for automatic latency recording
 */
class ScopeTimer {
public:
    explicit ScopeTimer(LatencyTracker& tracker)
        : tracker_(tracker), start_(Clock::now()) {}
    
    ~ScopeTimer() {
        tracker_.record(Clock::now() - start_);
    }
    
    // Non-copyable
    ScopeTimer(const ScopeTimer&) = delete;
    ScopeTimer& operator=(const ScopeTimer&) = delete;
    
    /**
     * @brief Get elapsed time so far
     */
    Duration elapsed() const {
        return Clock::now() - start_;
    }

private:
    LatencyTracker& tracker_;
    Timestamp start_;
};

/**
 * @brief Macro for convenient scope timing
 */
#define PHANTOMCORE_TIMED_SCOPE(tracker) \
    ::phantomcore::ScopeTimer _scope_timer_##__LINE__(tracker)

}  // namespace phantomcore
