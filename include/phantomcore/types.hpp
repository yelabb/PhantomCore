#pragma once

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

#include "aligned_allocator.hpp"  // For aligned::allocate/deallocate

namespace phantomcore {

// ============================================================================
// Timing Types
// ============================================================================
using Clock = std::chrono::high_resolution_clock;
using Timestamp = Clock::time_point;
using Duration = std::chrono::nanoseconds;

/// Converts duration to microseconds (double precision)
inline double to_microseconds(Duration d) {
    return std::chrono::duration<double, std::micro>(d).count();
}

/// Converts duration to milliseconds (double precision)
inline double to_milliseconds(Duration d) {
    return std::chrono::duration<double, std::milli>(d).count();
}

// ============================================================================
// Channel Configuration - Runtime Flexibility for Different Hardware
// ============================================================================

/**
 * @brief Known neural recording hardware configurations
 * 
 * Pre-defined configurations for common systems. Custom configurations
 * are also supported via ChannelConfig::custom().
 */
enum class HardwarePreset {
    UtahArray96,      // Utah array: 96 channels
    UtahArray128,     // Utah array: 128 channels  
    MCMaze142,        // MC_Maze dataset: 142 channels
    Neuropixels384,   // Neuropixels 1.0: 384 channels
    Neuropixels960,   // Neuropixels 2.0: 960 channels
    Custom            // User-defined channel count
};

/**
 * @brief Neural channel configuration
 * 
 * Abstracts the number of channels to support different hardware setups
 * without recompilation. All components use this config at runtime.
 */
struct ChannelConfig {
    size_t num_channels = 142;        // Default: MC_Maze compatibility
    HardwarePreset preset = HardwarePreset::MCMaze142;
    std::string name = "MC_Maze";
    
    // Factory methods for common configurations
    static ChannelConfig utah_array_96() {
        return {96, HardwarePreset::UtahArray96, "Utah Array 96ch"};
    }
    
    static ChannelConfig utah_array_128() {
        return {128, HardwarePreset::UtahArray128, "Utah Array 128ch"};
    }
    
    static ChannelConfig mc_maze() {
        return {142, HardwarePreset::MCMaze142, "MC_Maze Dataset"};
    }
    
    static ChannelConfig neuropixels() {
        return {384, HardwarePreset::Neuropixels384, "Neuropixels 1.0"};
    }
    
    static ChannelConfig neuropixels_2() {
        return {960, HardwarePreset::Neuropixels960, "Neuropixels 2.0"};
    }
    
    static ChannelConfig custom(size_t channels, const std::string& name = "Custom") {
        return {channels, HardwarePreset::Custom, name};
    }
    
    // Validation
    bool is_valid() const { return num_channels > 0 && num_channels <= 4096; }
};

/// Default channel count for backward compatibility (deprecated - use ChannelConfig)
[[deprecated("Use ChannelConfig for runtime channel configuration")]]
constexpr size_t NUM_CHANNELS = 142;

/// Streaming rate in Hz
constexpr double STREAM_RATE_HZ = 40.0;

/// Sample period in microseconds
constexpr double SAMPLE_PERIOD_US = 1'000'000.0 / STREAM_RATE_HZ;

// ============================================================================
// Dynamic Spike Data - Runtime Channel Count
// ============================================================================

// Note: aligned::allocate/deallocate are defined in aligned_allocator.hpp

/**
 * @brief Dynamic spike count array with 32-byte alignment
 * 
 * Replaces fixed-size SpikeCountArray with runtime-sized allocation.
 * Maintains SIMD alignment for AVX2 operations.
 */
class SpikeData {
public:
    SpikeData() = default;
    
    explicit SpikeData(size_t num_channels) 
        : num_channels_(num_channels)
        , data_(static_cast<float*>(aligned::allocate(num_channels * sizeof(float), 32)))
    {
        if (!data_) {
            throw std::bad_alloc();
        }
        std::fill_n(data_, num_channels, 0.0f);
    }
    
    explicit SpikeData(const ChannelConfig& config) 
        : SpikeData(config.num_channels) {}
    
    ~SpikeData() {
        if (data_) {
            aligned::deallocate(data_);
        }
    }
    
    // Move semantics
    SpikeData(SpikeData&& other) noexcept 
        : num_channels_(other.num_channels_)
        , data_(other.data_)
    {
        other.num_channels_ = 0;
        other.data_ = nullptr;
    }
    
    SpikeData& operator=(SpikeData&& other) noexcept {
        if (this != &other) {
            if (data_) aligned::deallocate(data_);
            num_channels_ = other.num_channels_;
            data_ = other.data_;
            other.num_channels_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }
    
    // Copy semantics
    SpikeData(const SpikeData& other) 
        : num_channels_(other.num_channels_)
        , data_(static_cast<float*>(aligned::allocate(other.num_channels_ * sizeof(float), 32)))
    {
        if (!data_ && num_channels_ > 0) throw std::bad_alloc();
        std::copy_n(other.data_, num_channels_, data_);
    }
    
    SpikeData& operator=(const SpikeData& other) {
        if (this != &other) {
            SpikeData tmp(other);
            std::swap(num_channels_, tmp.num_channels_);
            std::swap(data_, tmp.data_);
        }
        return *this;
    }
    
    // Element access
    float& operator[](size_t i) { return data_[i]; }
    const float& operator[](size_t i) const { return data_[i]; }
    
    float* data() noexcept { return data_; }
    const float* data() const noexcept { return data_; }
    
    size_t size() const noexcept { return num_channels_; }
    bool empty() const noexcept { return num_channels_ == 0; }
    
    // STL-like iterators
    float* begin() noexcept { return data_; }
    float* end() noexcept { return data_ + num_channels_; }
    const float* begin() const noexcept { return data_; }
    const float* end() const noexcept { return data_ + num_channels_; }
    
    // Span conversion for API compatibility
    std::span<float> span() noexcept { return {data_, num_channels_}; }
    std::span<const float> span() const noexcept { return {data_, num_channels_}; }
    
    // Fill with value
    void fill(float value) noexcept {
        std::fill_n(data_, num_channels_, value);
    }
    
    // Reset to zeros
    void zero() noexcept { fill(0.0f); }

private:
    size_t num_channels_ = 0;
    float* data_ = nullptr;
};

/**
 * @brief Legacy fixed-size spike count array (deprecated)
 * 
 * Kept for backward compatibility with existing code.
 * New code should use SpikeData instead.
 */
using SpikeCountArray = std::array<int32_t, 142>;  // Fixed for legacy compatibility

/// Aligned spike count array for AVX2 (256-bit = 8 floats) - Legacy
struct alignas(32) AlignedSpikeData {
    std::array<float, 142> counts;  // Fixed for legacy compatibility
    
    float& operator[](size_t i) { return counts[i]; }
    const float& operator[](size_t i) const { return counts[i]; }
    float* data() { return counts.data(); }
    const float* data() const { return counts.data(); }
    static constexpr size_t size() { return 142; }
};

/// 2D position (cursor/target)
struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
    
    Vec2 operator+(const Vec2& other) const { return {x + other.x, y + other.y}; }
    Vec2 operator-(const Vec2& other) const { return {x - other.x, y - other.y}; }
    Vec2 operator*(float s) const { return {x * s, y * s}; }
    
    float norm() const { return std::sqrt(x * x + y * y); }
    float norm_sq() const { return x * x + y * y; }
};

/// 2D velocity
struct Velocity2D {
    float vx = 0.0f;
    float vy = 0.0f;
};

/// 4D state vector (position + velocity)
struct Vec4 {
    float x = 0.0f;
    float y = 0.0f;
    float vx = 0.0f;
    float vy = 0.0f;
    
    Vec4 operator+(const Vec4& other) const { 
        return {x + other.x, y + other.y, vx + other.vx, vy + other.vy}; 
    }
    Vec4 operator-(const Vec4& other) const { 
        return {x - other.x, y - other.y, vx - other.vx, vy - other.vy}; 
    }
    Vec4 operator*(float s) const { 
        return {x * s, y * s, vx * s, vy * s}; 
    }
    
    float norm() const { return std::sqrt(x*x + y*y + vx*vx + vy*vy); }
    
    Vec2 position() const { return {x, y}; }
    Vec2 velocity() const { return {vx, vy}; }
};

/// Complete kinematics state
struct Kinematics {
    Vec2 position;
    Velocity2D velocity;
    
    float x() const { return position.x; }
    float y() const { return position.y; }
    float vx() const { return velocity.vx; }
    float vy() const { return velocity.vy; }
};

/// Target intention data
struct Intention {
    int32_t target_id = -1;
    Vec2 target_position;
    float distance = 0.0f;
};

/// Single neural data packet (matches PhantomLink format)
struct NeuralPacket {
    uint64_t sequence = 0;
    uint64_t trial_id = 0;
    double timestamp = 0.0;
    
    SpikeCountArray spike_counts;
    Kinematics kinematics;
    Intention intention;
    
    Timestamp received_at;  // Local receive timestamp for latency tracking
};

/// Decoder output
struct DecoderOutput {
    Vec2 position;
    Velocity2D velocity;
    float confidence = 1.0f;
    Duration processing_time;
};

// ============================================================================
// Performance Metrics
// ============================================================================

struct LatencyStats {
    double min_us = 0.0;
    double max_us = 0.0;
    double mean_us = 0.0;
    double std_us = 0.0;
    double p50_us = 0.0;
    double p95_us = 0.0;
    double p99_us = 0.0;
    size_t sample_count = 0;
};

struct PerformanceMetrics {
    LatencyStats decode_latency;
    LatencyStats network_latency;
    LatencyStats total_latency;
    
    double packets_per_second = 0.0;
    uint64_t total_packets = 0;
    uint64_t dropped_packets = 0;
    
    double accuracy = 0.0;      // Decoder accuracy (if ground truth available)
    double mean_error = 0.0;    // Mean position error
};

// ============================================================================
// Spike Detection Types
// ============================================================================

/// Detected spike event
struct SpikeEvent {
    uint32_t channel;
    double timestamp;
    float amplitude;
    uint32_t cluster_id;  // For sorted spikes
};

/// Spike detection parameters
struct SpikeDetectorConfig {
    float threshold_std = -4.0f;         // Threshold in standard deviations
    size_t refractory_samples = 32;      // Refractory period in samples
    size_t pre_samples = 16;             // Samples before threshold crossing
    size_t post_samples = 32;            // Samples after threshold crossing
    bool use_adaptive_threshold = true;   // Adapt threshold over time
    double adaptation_rate = 0.001;       // Threshold adaptation rate
    
    // DSP Bandpass Filter Configuration
    bool use_bandpass_filter = true;      // CRITICAL: Filter before detection
    float sample_rate = 30000.0f;         // Hz (typical neural recording rate)
    float bandpass_low = 300.0f;          // Hz (removes LFP, motion artifacts)
    float bandpass_high = 3000.0f;        // Hz (removes high-frequency noise)
    uint8_t filter_order = 4;             // Butterworth order (2 or 4)
};

/// Waveform snippet for spike sorting
struct Waveform {
    std::array<float, 48> samples;  // pre + post samples
    uint32_t channel;
    double timestamp;
};

// ============================================================================
// Connection Types
// ============================================================================

enum class ConnectionState {
    Disconnected,
    Connecting,
    Connected,
    Reconnecting,
    Error
};

struct ConnectionConfig {
    std::string url = "ws://localhost:8000/stream/binary/";
    std::string session_code = "";
    uint32_t reconnect_delay_ms = 1000;
    uint32_t max_reconnect_attempts = 5;
    bool auto_reconnect = true;
};

}  // namespace phantomcore
