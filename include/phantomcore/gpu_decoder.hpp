#pragma once

#include "types.hpp"
#include "kalman_decoder.hpp"
#include <memory>
#include <span>
#include <optional>
#include <expected>

namespace phantomcore {

// ============================================================================
// GPU Backend Configuration
// ============================================================================

/**
 * @brief GPU compute backends supported by PhantomCore
 */
enum class GPUBackend {
    None,       // CPU fallback
    CUDA,       // NVIDIA CUDA
    Metal,      // Apple Metal (future)
    Vulkan      // Vulkan Compute (future)
};

/**
 * @brief GPU device information
 */
struct GPUDeviceInfo {
    int device_id = -1;
    std::string name;
    size_t total_memory_mb = 0;
    size_t free_memory_mb = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int max_threads_per_block = 0;
    int warp_size = 0;
    bool supports_fp16 = false;
    bool supports_tensor_cores = false;
};

/**
 * @brief GPU decoder execution mode
 */
enum class GPUExecutionMode {
    Synchronous,    // Block until complete (lowest latency for single decode)
    Asynchronous,   // Non-blocking, use get_result() to retrieve
    Pipelined       // Overlap compute with data transfer (highest throughput)
};

/**
 * @brief Error codes for GPU operations
 */
enum class GPUError {
    Success = 0,
    NotInitialized,
    DeviceNotFound,
    OutOfMemory,
    KernelLaunchFailed,
    InvalidInput,
    Timeout,
    CUDAError,
    UnsupportedOperation
};

// ============================================================================
// GPU Decoder Interface
// ============================================================================

/**
 * @brief GPU-accelerated neural decoder for high-channel-count systems
 * 
 * Provides CUDA-accelerated Kalman decoding optimized for:
 * - Neuropixels 2.0 (960 channels)
 * - Multi-probe configurations (2000-4000+ channels)
 * - Batch decoding for offline analysis
 * 
 * Architecture:
 * ```
 *   Host Memory          Device Memory           Compute
 *   ┌─────────┐         ┌─────────────┐         ┌─────────┐
 *   │ Spikes  │ ──DMA──▶│ d_spikes    │ ──────▶ │ Kernel  │
 *   └─────────┘  async  └─────────────┘         └────┬────┘
 *                                                    │
 *   ┌─────────┐         ┌─────────────┐              │
 *   │ Output  │ ◀──DMA──│ d_output    │ ◀────────────┘
 *   └─────────┘  async  └─────────────┘
 * ```
 * 
 * Latency characteristics:
 * - Synchronous mode: ~8-12μs for 960 channels
 * - Pipelined mode: ~4-6μs steady-state (overlapped transfers)
 * 
 * @note Falls back to CPU if CUDA unavailable or for small channel counts
 */
class GPUDecoder {
public:
    struct Config {
        /// Channel configuration
        ChannelConfig channel_config = ChannelConfig::neuropixels();
        
        /// GPU device to use (-1 = auto-select best)
        int device_id = -1;
        
        /// Execution mode
        GPUExecutionMode execution_mode = GPUExecutionMode::Synchronous;
        
        /// Enable automatic CPU fallback
        bool enable_cpu_fallback = true;
        
        /// Channel threshold for GPU activation (use CPU below this)
        size_t gpu_channel_threshold = 256;
        
        /// Pre-allocate buffers for this many samples (batch mode)
        size_t batch_capacity = 1;
        
        /// Use FP16 for intermediate computations (faster on Tensor Cores)
        bool use_fp16 = false;
        
        /// CUDA stream priority (-1 = highest, 0 = default)
        int stream_priority = -1;
        
        /// Kalman decoder configuration
        KalmanDecoder::Config kalman_config;
        
        Config() = default;
    };
    
    explicit GPUDecoder(const Config& config);
    GPUDecoder();
    ~GPUDecoder();
    
    // Non-copyable, movable
    GPUDecoder(const GPUDecoder&) = delete;
    GPUDecoder& operator=(const GPUDecoder&) = delete;
    GPUDecoder(GPUDecoder&&) noexcept;
    GPUDecoder& operator=(GPUDecoder&&) noexcept;
    
    // ========================================================================
    // Initialization & Status
    // ========================================================================
    
    /**
     * @brief Check if GPU acceleration is available
     */
    static bool is_gpu_available();
    
    /**
     * @brief Get list of available GPU devices
     */
    static std::vector<GPUDeviceInfo> get_available_devices();
    
    /**
     * @brief Get currently active GPU backend
     */
    GPUBackend active_backend() const;
    
    /**
     * @brief Get current device info (if GPU active)
     */
    std::optional<GPUDeviceInfo> device_info() const;
    
    /**
     * @brief Check if currently using GPU or CPU fallback
     */
    bool is_using_gpu() const;
    
    // ========================================================================
    // Decoding Interface
    // ========================================================================
    
    /**
     * @brief Decode spike data to kinematics
     * 
     * For synchronous mode, returns result immediately.
     * For async/pipelined, queues the decode and returns previous result.
     * 
     * @param spike_data Input spike counts
     * @return Decoded kinematics or error
     */
    std::expected<DecoderOutput, GPUError> decode(const SpikeData& spike_data);
    
    /**
     * @brief Decode with raw span (for zero-copy from ring buffer)
     */
    std::expected<DecoderOutput, GPUError> decode(std::span<const float> spike_counts);
    
    /**
     * @brief Batch decode multiple samples (offline analysis)
     * 
     * Optimized for throughput over latency.
     * 
     * @param spike_batch Matrix [num_samples x num_channels]
     * @return Vector of decoded outputs
     */
    std::expected<std::vector<DecoderOutput>, GPUError> decode_batch(
        std::span<const float> spike_batch,
        size_t num_samples
    );
    
    /**
     * @brief Predict without measurement (async-friendly)
     */
    std::expected<DecoderOutput, GPUError> predict();
    
    /**
     * @brief Get result from async decode (non-blocking)
     * @return Result if ready, nullopt if still processing
     */
    std::optional<std::expected<DecoderOutput, GPUError>> try_get_result();
    
    /**
     * @brief Wait for async decode to complete
     * @param timeout_us Maximum wait time in microseconds (0 = infinite)
     * @return Result or timeout error
     */
    std::expected<DecoderOutput, GPUError> wait_for_result(uint64_t timeout_us = 0);
    
    // ========================================================================
    // Calibration
    // ========================================================================
    
    /**
     * @brief Calibrate decoder on GPU (faster for large datasets)
     * 
     * @param neural_data Training data [num_samples x num_channels]
     * @param kinematics Ground truth [num_samples x 4]
     * @return Calibration result
     */
    KalmanDecoder::CalibrationResult calibrate(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& kinematics
    );
    
    /**
     * @brief Load pre-trained weights
     */
    void load_weights(std::span<const float> weights);
    
    /**
     * @brief Transfer calibrated CPU decoder to GPU
     */
    void transfer_from_cpu(const KalmanDecoder& cpu_decoder);
    
    // ========================================================================
    // State Management
    // ========================================================================
    
    /**
     * @brief Reset decoder state
     */
    void reset();
    
    /**
     * @brief Synchronize GPU operations (wait for all pending work)
     */
    void synchronize();
    
    /**
     * @brief Get current state estimate
     */
    KalmanDecoder::StateVector get_state() const;
    
    // ========================================================================
    // Performance Monitoring
    // ========================================================================
    
    struct PerformanceStats {
        uint64_t total_decodes = 0;
        uint64_t gpu_decodes = 0;
        uint64_t cpu_fallback_decodes = 0;
        
        LatencyStats decode_latency;
        LatencyStats h2d_transfer_latency;  // Host to Device
        LatencyStats d2h_transfer_latency;  // Device to Host
        LatencyStats kernel_latency;
        
        size_t peak_memory_usage_bytes = 0;
    };
    
    PerformanceStats get_stats() const;
    void reset_stats();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

// ============================================================================
// Multi-Probe Decoder (for multiple simultaneous probes)
// ============================================================================

/**
 * @brief Coordinates decoding across multiple neural probes
 * 
 * For multi-probe setups (e.g., 4x Neuropixels = 3840 channels), this class:
 * - Manages per-probe GPU streams for parallel execution
 * - Fuses outputs from multiple probes into unified kinematics
 * - Handles probe synchronization and timing alignment
 */
class MultiProbeDecoder {
public:
    struct ProbeConfig {
        ChannelConfig channel_config;
        std::string probe_name;
        int device_id = -1;  // -1 = share device with other probes
        
        /// Weight for output fusion (0.0-1.0)
        float fusion_weight = 1.0f;
    };
    
    struct Config {
        std::vector<ProbeConfig> probes;
        
        /// Fusion strategy for combining probe outputs
        enum class FusionStrategy {
            WeightedAverage,   // Simple weighted average
            KalmanFusion,      // Optimal fusion via Kalman
            SelectBest         // Use probe with lowest uncertainty
        };
        FusionStrategy fusion = FusionStrategy::KalmanFusion;
        
        /// Maximum timing skew between probes (μs)
        double max_timing_skew_us = 100.0;
    };
    
    explicit MultiProbeDecoder(const Config& config);
    ~MultiProbeDecoder();
    
    // Non-copyable, movable
    MultiProbeDecoder(const MultiProbeDecoder&) = delete;
    MultiProbeDecoder& operator=(const MultiProbeDecoder&) = delete;
    MultiProbeDecoder(MultiProbeDecoder&&) noexcept;
    MultiProbeDecoder& operator=(MultiProbeDecoder&&) noexcept;
    
    /**
     * @brief Decode from all probes simultaneously
     * @param probe_data Vector of spike data, one per probe
     * @return Fused decoder output
     */
    std::expected<DecoderOutput, GPUError> decode(
        const std::vector<SpikeData>& probe_data
    );
    
    /**
     * @brief Get number of configured probes
     */
    size_t num_probes() const;
    
    /**
     * @brief Get total channel count across all probes
     */
    size_t total_channels() const;
    
    /**
     * @brief Calibrate all probes
     */
    void calibrate(
        const std::vector<Eigen::MatrixXf>& neural_data,
        const Eigen::MatrixXf& kinematics
    );
    
    /**
     * @brief Reset all probe decoders
     */
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert GPU error to string
 */
const char* gpu_error_string(GPUError error);

/**
 * @brief Check CUDA availability at runtime
 */
bool cuda_available();

/**
 * @brief Get CUDA driver version
 */
std::optional<std::pair<int, int>> cuda_version();

}  // namespace phantomcore
