#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

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
// Neural Data Types
// ============================================================================

/// Number of neural channels (matches MC_Maze dataset)
constexpr size_t NUM_CHANNELS = 142;

/// Streaming rate in Hz
constexpr double STREAM_RATE_HZ = 40.0;

/// Sample period in microseconds
constexpr double SAMPLE_PERIOD_US = 1'000'000.0 / STREAM_RATE_HZ;

/// Fixed-size spike count array for SIMD optimization
using SpikeCountArray = std::array<int32_t, NUM_CHANNELS>;

/// Aligned spike count array for AVX2 (256-bit = 8 floats)
struct alignas(32) AlignedSpikeData {
    std::array<float, NUM_CHANNELS> counts;
    
    float& operator[](size_t i) { return counts[i]; }
    const float& operator[](size_t i) const { return counts[i]; }
    float* data() { return counts.data(); }
    const float* data() const { return counts.data(); }
    static constexpr size_t size() { return NUM_CHANNELS; }
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
