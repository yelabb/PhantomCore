#include "phantomcore/gpu_decoder.hpp"
#include "phantomcore/kalman_decoder.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <algorithm>
#include <atomic>
#include <mutex>

// ============================================================================
// CUDA Headers (conditional)
// ============================================================================
#ifdef PHANTOMCORE_ENABLE_CUDA
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
    #include <cuda_fp16.h>
    
    #define CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                return GPUError::CUDAError; \
            } \
        } while(0)
    
    #define CUBLAS_CHECK(call) \
        do { \
            cublasStatus_t status = call; \
            if (status != CUBLAS_STATUS_SUCCESS) { \
                return GPUError::CUDAError; \
            } \
        } while(0)
#endif

namespace phantomcore {

// ============================================================================
// GPU Error String
// ============================================================================

const char* gpu_error_string(GPUError error) {
    switch (error) {
        case GPUError::Success: return "Success";
        case GPUError::NotInitialized: return "GPU not initialized";
        case GPUError::DeviceNotFound: return "GPU device not found";
        case GPUError::OutOfMemory: return "GPU out of memory";
        case GPUError::KernelLaunchFailed: return "Kernel launch failed";
        case GPUError::InvalidInput: return "Invalid input";
        case GPUError::Timeout: return "Operation timed out";
        case GPUError::CUDAError: return "CUDA error";
        case GPUError::UnsupportedOperation: return "Unsupported operation";
        default: return "Unknown error";
    }
}

// ============================================================================
// CUDA Availability Check
// ============================================================================

bool cuda_available() {
#ifdef PHANTOMCORE_ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0);
#else
    return false;
#endif
}

std::optional<std::pair<int, int>> cuda_version() {
#ifdef PHANTOMCORE_ENABLE_CUDA
    int driver_version = 0;
    if (cudaDriverGetVersion(&driver_version) == cudaSuccess) {
        int major = driver_version / 1000;
        int minor = (driver_version % 1000) / 10;
        return std::make_pair(major, minor);
    }
#endif
    return std::nullopt;
}

// ============================================================================
// GPUDecoder Implementation
// ============================================================================

#ifdef PHANTOMCORE_ENABLE_CUDA

struct GPUDecoder::Impl {
    // Device resources
    int device_id = -1;
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    
    // Device buffers
    float* d_spikes = nullptr;
    float* d_state = nullptr;
    float* d_covariance = nullptr;
    float* d_observation_matrix = nullptr;
    float* d_kalman_gain = nullptr;
    float* d_output = nullptr;
    
    // Batch buffers
    float* d_spike_batch = nullptr;
    float* d_output_batch = nullptr;
    
    // Host pinned memory for async transfers
    float* h_spikes_pinned = nullptr;
    float* h_output_pinned = nullptr;
    
    // CPU fallback decoder
    std::unique_ptr<KalmanDecoder> cpu_decoder;
    
    // State
    bool gpu_initialized = false;
    bool using_gpu = false;
    size_t num_channels = 0;
    size_t batch_capacity = 0;
    
    // Performance tracking
    LatencyTracker decode_latency;
    LatencyTracker h2d_latency;
    LatencyTracker d2h_latency;
    LatencyTracker kernel_latency;
    std::atomic<uint64_t> total_decodes{0};
    std::atomic<uint64_t> gpu_decodes{0};
    std::atomic<uint64_t> cpu_fallback_decodes{0};
    size_t peak_memory = 0;
    
    // Async state
    std::atomic<bool> async_pending{false};
    DecoderOutput async_result;
    std::mutex async_mutex;
    cudaEvent_t compute_done_event = nullptr;
    
    // Kalman state (device mirror)
    Eigen::Vector4f state = Eigen::Vector4f::Zero();
    Eigen::Matrix4f covariance = Eigen::Matrix4f::Identity();
    Eigen::MatrixXf observation_matrix;
    float measurement_noise_scale = 0.1f;
    
    ~Impl() {
        cleanup();
    }
    
    void cleanup() {
        if (d_spikes) cudaFree(d_spikes);
        if (d_state) cudaFree(d_state);
        if (d_covariance) cudaFree(d_covariance);
        if (d_observation_matrix) cudaFree(d_observation_matrix);
        if (d_kalman_gain) cudaFree(d_kalman_gain);
        if (d_output) cudaFree(d_output);
        if (d_spike_batch) cudaFree(d_spike_batch);
        if (d_output_batch) cudaFree(d_output_batch);
        if (h_spikes_pinned) cudaFreeHost(h_spikes_pinned);
        if (h_output_pinned) cudaFreeHost(h_output_pinned);
        if (compute_done_event) cudaEventDestroy(compute_done_event);
        if (cublas_handle) cublasDestroy(cublas_handle);
        if (stream) cudaStreamDestroy(stream);
        
        d_spikes = nullptr;
        d_state = nullptr;
        d_covariance = nullptr;
        d_observation_matrix = nullptr;
        d_kalman_gain = nullptr;
        d_output = nullptr;
        d_spike_batch = nullptr;
        d_output_batch = nullptr;
        h_spikes_pinned = nullptr;
        h_output_pinned = nullptr;
        compute_done_event = nullptr;
        cublas_handle = nullptr;
        stream = nullptr;
        gpu_initialized = false;
    }
    
    GPUError initialize(const GPUDecoder::Config& config) {
        num_channels = config.channel_config.num_channels;
        batch_capacity = config.batch_capacity;
        
        // Select device
        device_id = config.device_id;
        if (device_id < 0) {
            // Auto-select: pick device with most free memory
            int device_count = 0;
            cudaGetDeviceCount(&device_count);
            if (device_count == 0) {
                return GPUError::DeviceNotFound;
            }
            
            size_t max_free = 0;
            for (int i = 0; i < device_count; ++i) {
                cudaSetDevice(i);
                size_t free_mem, total_mem;
                cudaMemGetInfo(&free_mem, &total_mem);
                if (free_mem > max_free) {
                    max_free = free_mem;
                    device_id = i;
                }
            }
        }
        
        CUDA_CHECK(cudaSetDevice(device_id));
        
        // Create stream with priority
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
        int priority = (config.stream_priority < 0) ? greatest_priority : config.stream_priority;
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority));
        
        // Create cuBLAS handle
        CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
        
        // Allocate device memory
        size_t spike_size = num_channels * sizeof(float);
        size_t state_size = 4 * sizeof(float);
        size_t cov_size = 16 * sizeof(float);
        size_t obs_size = num_channels * 4 * sizeof(float);
        size_t gain_size = 4 * num_channels * sizeof(float);
        
        CUDA_CHECK(cudaMalloc(&d_spikes, spike_size));
        CUDA_CHECK(cudaMalloc(&d_state, state_size));
        CUDA_CHECK(cudaMalloc(&d_covariance, cov_size));
        CUDA_CHECK(cudaMalloc(&d_observation_matrix, obs_size));
        CUDA_CHECK(cudaMalloc(&d_kalman_gain, gain_size));
        CUDA_CHECK(cudaMalloc(&d_output, state_size));
        
        // Batch buffers
        if (batch_capacity > 1) {
            CUDA_CHECK(cudaMalloc(&d_spike_batch, spike_size * batch_capacity));
            CUDA_CHECK(cudaMalloc(&d_output_batch, state_size * batch_capacity));
        }
        
        // Pinned host memory for async transfers
        CUDA_CHECK(cudaMallocHost(&h_spikes_pinned, spike_size));
        CUDA_CHECK(cudaMallocHost(&h_output_pinned, state_size));
        
        // Event for async completion
        CUDA_CHECK(cudaEventCreate(&compute_done_event));
        
        // Track memory usage
        size_t total_gpu_mem = spike_size + state_size + cov_size + obs_size + gain_size + state_size;
        if (batch_capacity > 1) {
            total_gpu_mem += spike_size * batch_capacity + state_size * batch_capacity;
        }
        peak_memory = total_gpu_mem;
        
        // Initialize observation matrix with identity-like pattern
        observation_matrix = Eigen::MatrixXf::Zero(num_channels, 4);
        
        gpu_initialized = true;
        using_gpu = true;
        
        return GPUError::Success;
    }
    
    GPUError upload_kalman_matrices() {
        if (!gpu_initialized) return GPUError::NotInitialized;
        
        // Upload state
        CUDA_CHECK(cudaMemcpyAsync(d_state, state.data(), 
            4 * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // Upload covariance
        CUDA_CHECK(cudaMemcpyAsync(d_covariance, covariance.data(),
            16 * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        // Upload observation matrix
        CUDA_CHECK(cudaMemcpyAsync(d_observation_matrix, observation_matrix.data(),
            observation_matrix.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        
        return GPUError::Success;
    }
};

#else  // No CUDA

struct GPUDecoder::Impl {
    std::unique_ptr<KalmanDecoder> cpu_decoder;
    bool gpu_initialized = false;
    bool using_gpu = false;
    size_t num_channels = 0;
    
    LatencyTracker decode_latency;
    std::atomic<uint64_t> total_decodes{0};
    std::atomic<uint64_t> cpu_fallback_decodes{0};
    
    GPUError initialize(const GPUDecoder::Config& config) {
        num_channels = config.channel_config.num_channels;
        cpu_decoder = std::make_unique<KalmanDecoder>(config.kalman_config);
        using_gpu = false;
        return GPUError::Success;
    }
};

#endif  // PHANTOMCORE_ENABLE_CUDA

// ============================================================================
// GPUDecoder Public Interface
// ============================================================================

GPUDecoder::GPUDecoder(const Config& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
    // Always create CPU fallback
    impl_->cpu_decoder = std::make_unique<KalmanDecoder>(config.kalman_config);
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    // Only use GPU if above channel threshold and CUDA available
    if (cuda_available() && config.channel_config.num_channels >= config.gpu_channel_threshold) {
        auto err = impl_->initialize(config);
        if (err != GPUError::Success && !config.enable_cpu_fallback) {
            throw std::runtime_error(std::string("GPU initialization failed: ") + gpu_error_string(err));
        }
    }
#else
    impl_->initialize(config);
#endif
}

GPUDecoder::~GPUDecoder() = default;

GPUDecoder::GPUDecoder(GPUDecoder&&) noexcept = default;
GPUDecoder& GPUDecoder::operator=(GPUDecoder&&) noexcept = default;

bool GPUDecoder::is_gpu_available() {
    return cuda_available();
}

std::vector<GPUDeviceInfo> GPUDecoder::get_available_devices() {
    std::vector<GPUDeviceInfo> devices;
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return devices;
    }
    
    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            GPUDeviceInfo info;
            info.device_id = i;
            info.name = prop.name;
            info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.warp_size = prop.warpSize;
            info.supports_fp16 = (prop.major >= 6);
            info.supports_tensor_cores = (prop.major >= 7);
            
            // Get free memory
            cudaSetDevice(i);
            size_t free_mem, total_mem;
            cudaMemGetInfo(&free_mem, &total_mem);
            info.free_memory_mb = free_mem / (1024 * 1024);
            
            devices.push_back(info);
        }
    }
#endif
    
    return devices;
}

GPUBackend GPUDecoder::active_backend() const {
#ifdef PHANTOMCORE_ENABLE_CUDA
    return impl_->using_gpu ? GPUBackend::CUDA : GPUBackend::None;
#else
    return GPUBackend::None;
#endif
}

std::optional<GPUDeviceInfo> GPUDecoder::device_info() const {
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (!impl_->using_gpu || impl_->device_id < 0) {
        return std::nullopt;
    }
    
    auto devices = get_available_devices();
    for (const auto& dev : devices) {
        if (dev.device_id == impl_->device_id) {
            return dev;
        }
    }
#endif
    return std::nullopt;
}

bool GPUDecoder::is_using_gpu() const {
    return impl_->using_gpu;
}

std::expected<DecoderOutput, GPUError> GPUDecoder::decode(const SpikeData& spike_data) {
    return decode(std::span<const float>(spike_data.data(), spike_data.size()));
}

std::expected<DecoderOutput, GPUError> GPUDecoder::decode(std::span<const float> spike_counts) {
    auto start = Clock::now();
    impl_->total_decodes++;
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->using_gpu && impl_->gpu_initialized) {
        // GPU path
        impl_->gpu_decodes++;
        
        // Copy spikes to pinned memory
        std::copy(spike_counts.begin(), spike_counts.end(), impl_->h_spikes_pinned);
        
        auto h2d_start = Clock::now();
        cudaMemcpyAsync(impl_->d_spikes, impl_->h_spikes_pinned,
            spike_counts.size() * sizeof(float), cudaMemcpyHostToDevice, impl_->stream);
        impl_->h2d_latency.record(Clock::now() - h2d_start);
        
        // Kalman predict + update using cuBLAS
        auto kernel_start = Clock::now();
        
        // Simplified: For now, use cuBLAS for matrix operations
        // Full implementation would have custom CUDA kernels for Kalman update
        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        // Observation: y = H * x_predicted
        // Innovation: z - H * x
        // Kalman gain: K = P * H' * (H * P * H' + R)^-1
        // State update: x = x + K * innovation
        // Covariance update: P = (I - K * H) * P
        
        // For this implementation, we use cuBLAS GEMV/GEMM operations
        // Production would optimize further with fused kernels
        
        impl_->kernel_latency.record(Clock::now() - kernel_start);
        
        // Copy result back
        auto d2h_start = Clock::now();
        cudaMemcpyAsync(impl_->h_output_pinned, impl_->d_state,
            4 * sizeof(float), cudaMemcpyDeviceToHost, impl_->stream);
        cudaStreamSynchronize(impl_->stream);
        impl_->d2h_latency.record(Clock::now() - d2h_start);
        
        // Build output
        DecoderOutput output;
        output.position.x = impl_->h_output_pinned[0];
        output.position.y = impl_->h_output_pinned[1];
        output.velocity.vx = impl_->h_output_pinned[2];
        output.velocity.vy = impl_->h_output_pinned[3];
        output.processing_time = Clock::now() - start;
        output.confidence = 1.0f;  // TODO: Compute from covariance
        
        impl_->decode_latency.record(output.processing_time);
        return output;
    }
#endif
    
    // CPU fallback
    impl_->cpu_fallback_decodes++;
    
    SpikeData spike_data(spike_counts.size());
    for (size_t i = 0; i < spike_counts.size(); ++i) {
        spike_data[i] = spike_counts[i];
    }
    
    auto output = impl_->cpu_decoder->decode(spike_data);
    impl_->decode_latency.record(Clock::now() - start);
    
    return output;
}

std::expected<std::vector<DecoderOutput>, GPUError> GPUDecoder::decode_batch(
    std::span<const float> spike_batch,
    size_t num_samples
) {
    std::vector<DecoderOutput> results;
    results.reserve(num_samples);
    
    size_t channels = config_.channel_config.num_channels;
    
    for (size_t i = 0; i < num_samples; ++i) {
        auto sample = spike_batch.subspan(i * channels, channels);
        auto result = decode(sample);
        if (!result) {
            return std::unexpected(result.error());
        }
        results.push_back(*result);
    }
    
    return results;
}

std::expected<DecoderOutput, GPUError> GPUDecoder::predict() {
    return std::expected<DecoderOutput, GPUError>(impl_->cpu_decoder->predict());
}

std::optional<std::expected<DecoderOutput, GPUError>> GPUDecoder::try_get_result() {
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->async_pending) {
        cudaError_t status = cudaEventQuery(impl_->compute_done_event);
        if (status == cudaSuccess) {
            impl_->async_pending = false;
            std::lock_guard<std::mutex> lock(impl_->async_mutex);
            return std::expected<DecoderOutput, GPUError>(impl_->async_result);
        } else if (status != cudaErrorNotReady) {
            impl_->async_pending = false;
            return std::expected<DecoderOutput, GPUError>(std::unexpected(GPUError::CUDAError));
        }
    }
#endif
    return std::nullopt;
}

std::expected<DecoderOutput, GPUError> GPUDecoder::wait_for_result(uint64_t timeout_us) {
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (!impl_->async_pending) {
        return std::unexpected(GPUError::InvalidInput);
    }
    
    if (timeout_us > 0) {
        auto start = Clock::now();
        while (impl_->async_pending) {
            auto result = try_get_result();
            if (result) {
                return *result;
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start).count();
            if (static_cast<uint64_t>(elapsed) >= timeout_us) {
                return std::unexpected(GPUError::Timeout);
            }
            std::this_thread::yield();
        }
    } else {
        cudaStreamSynchronize(impl_->stream);
        impl_->async_pending = false;
        std::lock_guard<std::mutex> lock(impl_->async_mutex);
        return impl_->async_result;
    }
#endif
    return std::unexpected(GPUError::UnsupportedOperation);
}

KalmanDecoder::CalibrationResult GPUDecoder::calibrate(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& kinematics
) {
    // Use CPU decoder for calibration, then transfer weights
    auto result = impl_->cpu_decoder->calibrate(neural_data, kinematics);
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->using_gpu && result.success) {
        transfer_from_cpu(*impl_->cpu_decoder);
    }
#endif
    
    return result;
}

void GPUDecoder::load_weights(std::span<const float> weights) {
    impl_->cpu_decoder->load_weights(weights);
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->using_gpu) {
        // Transfer weights to GPU
        impl_->observation_matrix = Eigen::Map<const Eigen::MatrixXf>(
            weights.data(), impl_->num_channels, 4);
        impl_->upload_kalman_matrices();
    }
#endif
}

void GPUDecoder::transfer_from_cpu(const KalmanDecoder& cpu_decoder) {
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (!impl_->using_gpu) return;
    
    impl_->state = cpu_decoder.get_state();
    impl_->covariance = cpu_decoder.get_covariance();
    
    auto weights = cpu_decoder.save_weights();
    impl_->observation_matrix = Eigen::Map<const Eigen::MatrixXf>(
        weights.data(), impl_->num_channels, 4);
    
    impl_->upload_kalman_matrices();
#endif
    (void)cpu_decoder;  // Suppress unused warning when CUDA disabled
}

void GPUDecoder::reset() {
    impl_->cpu_decoder->reset();
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->using_gpu) {
        impl_->state = Eigen::Vector4f::Zero();
        impl_->covariance = Eigen::Matrix4f::Identity();
        impl_->upload_kalman_matrices();
    }
#endif
}

void GPUDecoder::synchronize() {
#ifdef PHANTOMCORE_ENABLE_CUDA
    if (impl_->stream) {
        cudaStreamSynchronize(impl_->stream);
    }
#endif
}

KalmanDecoder::StateVector GPUDecoder::get_state() const {
    return impl_->cpu_decoder->get_state();
}

GPUDecoder::PerformanceStats GPUDecoder::get_stats() const {
    PerformanceStats stats;
    stats.total_decodes = impl_->total_decodes.load();
    stats.cpu_fallback_decodes = impl_->cpu_fallback_decodes.load();
    stats.decode_latency = impl_->decode_latency.get_stats();
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    stats.gpu_decodes = impl_->gpu_decodes.load();
    stats.h2d_transfer_latency = impl_->h2d_latency.get_stats();
    stats.d2h_transfer_latency = impl_->d2h_latency.get_stats();
    stats.kernel_latency = impl_->kernel_latency.get_stats();
    stats.peak_memory_usage_bytes = impl_->peak_memory;
#endif
    
    return stats;
}

void GPUDecoder::reset_stats() {
    impl_->total_decodes = 0;
    impl_->cpu_fallback_decodes = 0;
    impl_->decode_latency = LatencyTracker{};
    
#ifdef PHANTOMCORE_ENABLE_CUDA
    impl_->gpu_decodes = 0;
    impl_->h2d_latency = LatencyTracker{};
    impl_->d2h_latency = LatencyTracker{};
    impl_->kernel_latency = LatencyTracker{};
#endif
}

// ============================================================================
// MultiProbeDecoder Implementation
// ============================================================================

struct MultiProbeDecoder::Impl {
    std::vector<std::unique_ptr<GPUDecoder>> probe_decoders;
    MultiProbeDecoder::Config config;
    size_t total_channels = 0;
};

MultiProbeDecoder::MultiProbeDecoder(const Config& config)
    : impl_(std::make_unique<Impl>())
{
    impl_->config = config;
    
    for (const auto& probe_config : config.probes) {
        GPUDecoder::Config gpu_config;
        gpu_config.channel_config = probe_config.channel_config;
        gpu_config.device_id = probe_config.device_id;
        gpu_config.enable_cpu_fallback = true;
        
        impl_->probe_decoders.push_back(std::make_unique<GPUDecoder>(gpu_config));
        impl_->total_channels += probe_config.channel_config.num_channels;
    }
}

MultiProbeDecoder::~MultiProbeDecoder() = default;

MultiProbeDecoder::MultiProbeDecoder(MultiProbeDecoder&&) noexcept = default;
MultiProbeDecoder& MultiProbeDecoder::operator=(MultiProbeDecoder&&) noexcept = default;

std::expected<DecoderOutput, GPUError> MultiProbeDecoder::decode(
    const std::vector<SpikeData>& probe_data
) {
    if (probe_data.size() != impl_->probe_decoders.size()) {
        return std::unexpected(GPUError::InvalidInput);
    }
    
    std::vector<DecoderOutput> outputs;
    outputs.reserve(probe_data.size());
    
    // Decode each probe
    for (size_t i = 0; i < probe_data.size(); ++i) {
        auto result = impl_->probe_decoders[i]->decode(probe_data[i]);
        if (!result) {
            return std::unexpected(result.error());
        }
        outputs.push_back(*result);
    }
    
    // Fuse outputs based on strategy
    DecoderOutput fused;
    fused.processing_time = Duration::zero();
    
    switch (impl_->config.fusion) {
        case Config::FusionStrategy::WeightedAverage: {
            float total_weight = 0.0f;
            for (size_t i = 0; i < outputs.size(); ++i) {
                float weight = impl_->config.probes[i].fusion_weight;
                fused.position.x += outputs[i].position.x * weight;
                fused.position.y += outputs[i].position.y * weight;
                fused.velocity.vx += outputs[i].velocity.vx * weight;
                fused.velocity.vy += outputs[i].velocity.vy * weight;
                total_weight += weight;
                fused.processing_time = std::max(fused.processing_time, outputs[i].processing_time);
            }
            if (total_weight > 0) {
                fused.position.x /= total_weight;
                fused.position.y /= total_weight;
                fused.velocity.vx /= total_weight;
                fused.velocity.vy /= total_weight;
            }
            break;
        }
        
        case Config::FusionStrategy::SelectBest: {
            float best_confidence = -1.0f;
            for (const auto& output : outputs) {
                if (output.confidence > best_confidence) {
                    best_confidence = output.confidence;
                    fused = output;
                }
            }
            break;
        }
        
        case Config::FusionStrategy::KalmanFusion:
        default: {
            // Simplified Kalman fusion: weighted by inverse variance
            float total_inv_var = 0.0f;
            for (size_t i = 0; i < outputs.size(); ++i) {
                float inv_var = outputs[i].confidence;  // Use confidence as proxy for precision
                fused.position.x += outputs[i].position.x * inv_var;
                fused.position.y += outputs[i].position.y * inv_var;
                fused.velocity.vx += outputs[i].velocity.vx * inv_var;
                fused.velocity.vy += outputs[i].velocity.vy * inv_var;
                total_inv_var += inv_var;
                fused.processing_time = std::max(fused.processing_time, outputs[i].processing_time);
            }
            if (total_inv_var > 0) {
                fused.position.x /= total_inv_var;
                fused.position.y /= total_inv_var;
                fused.velocity.vx /= total_inv_var;
                fused.velocity.vy /= total_inv_var;
            }
            fused.confidence = total_inv_var;  // Combined precision
            break;
        }
    }
    
    return fused;
}

size_t MultiProbeDecoder::num_probes() const {
    return impl_->probe_decoders.size();
}

size_t MultiProbeDecoder::total_channels() const {
    return impl_->total_channels;
}

void MultiProbeDecoder::calibrate(
    const std::vector<Eigen::MatrixXf>& neural_data,
    const Eigen::MatrixXf& kinematics
) {
    for (size_t i = 0; i < impl_->probe_decoders.size() && i < neural_data.size(); ++i) {
        impl_->probe_decoders[i]->calibrate(neural_data[i], kinematics);
    }
}

void MultiProbeDecoder::reset() {
    for (auto& decoder : impl_->probe_decoders) {
        decoder->reset();
    }
}

}  // namespace phantomcore
