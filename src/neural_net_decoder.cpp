#include "phantomcore/neural_net_decoder.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <array>
#include <span>
#include <mutex>
#include <atomic>
#include <sstream>
#include <fstream>

// ============================================================================
// ONNX Runtime Headers (conditional)
// ============================================================================
#ifdef PHANTOMCORE_ENABLE_ONNX
    #include <onnxruntime_cxx_api.h>
    #ifdef PHANTOMCORE_ENABLE_CUDA
        #include <onnxruntime_cuda_provider_options.h>
    #endif
#endif

namespace phantomcore {

// ============================================================================
// Error Strings
// ============================================================================

const char* nn_error_string(NNError error) {
    switch (error) {
        case NNError::Success: return "Success";
        case NNError::ModelNotLoaded: return "Model not loaded";
        case NNError::InvalidModel: return "Invalid model";
        case NNError::InferenceFailed: return "Inference failed";
        case NNError::BackendNotAvailable: return "Backend not available";
        case NNError::ShapeMismatch: return "Shape mismatch";
        case NNError::OutOfMemory: return "Out of memory";
        case NNError::Timeout: return "Inference timeout";
        case NNError::FileNotFound: return "File not found";
        case NNError::UnsupportedArchitecture: return "Unsupported architecture";
        default: return "Unknown error";
    }
}

const char* nn_backend_string(NNBackend backend) {
    switch (backend) {
        case NNBackend::ONNX: return "ONNX Runtime";
        case NNBackend::TensorRT: return "TensorRT";
        case NNBackend::CoreML: return "CoreML";
        case NNBackend::OpenVINO: return "OpenVINO";
        case NNBackend::DirectML: return "DirectML";
        default: return "Unknown";
    }
}

const char* nn_architecture_string(NNArchitecture arch) {
    switch (arch) {
        case NNArchitecture::MLP: return "MLP";
        case NNArchitecture::LSTM: return "LSTM";
        case NNArchitecture::GRU: return "GRU";
        case NNArchitecture::TCN: return "TCN";
        case NNArchitecture::Transformer: return "Transformer";
        case NNArchitecture::Hybrid: return "Hybrid";
        case NNArchitecture::Custom: return "Custom";
        default: return "Unknown";
    }
}

// ============================================================================
// NeuralNetDecoder Implementation
// ============================================================================

struct NeuralNetDecoder::Impl {
#ifdef PHANTOMCORE_ENABLE_ONNX
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "PhantomCore"};
    std::unique_ptr<Ort::Session> session;
    Ort::SessionOptions session_options;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    // Pre-allocated tensors for inference
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    // Input/output names (stored as string to manage lifetime)
    std::vector<std::string> input_names_storage;
    std::vector<std::string> output_names_storage;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
#endif
    
    // Model info
    bool model_loaded = false;
    NNModelInfo model_info;
    
    // Sequence buffer for recurrent models (pre-allocated circular buffer)
    // PERF: Using fixed array + index instead of std::deque to avoid allocations in hot path
    static constexpr size_t MAX_SEQUENCE_CAPACITY = 64;  // Power of 2 for efficient modulo
    std::array<std::vector<float>, MAX_SEQUENCE_CAPACITY> sequence_buffer;
    size_t sequence_head = 0;      // Next write position
    size_t sequence_count = 0;     // Current number of valid samples
    size_t max_sequence_length = 20;
    bool sequence_buffer_initialized = false;
    
    void init_sequence_buffer(size_t num_channels) {
        if (!sequence_buffer_initialized) {
            for (auto& sample : sequence_buffer) {
                sample.resize(num_channels, 0.0f);
            }
            sequence_buffer_initialized = true;
        }
    }
    
    void push_to_sequence(std::span<const float> sample) {
        // Copy into pre-allocated slot (no allocation)
        auto& slot = sequence_buffer[sequence_head];
        std::copy(sample.begin(), sample.end(), slot.begin());
        
        sequence_head = (sequence_head + 1) % MAX_SEQUENCE_CAPACITY;
        if (sequence_count < max_sequence_length) {
            sequence_count++;
        }
    }
    
    void clear_sequence() {
        sequence_head = 0;
        sequence_count = 0;
    }
    
    // Iterator over valid sequence samples (oldest to newest)
    template<typename Func>
    void for_each_sequence_sample(Func&& func) const {
        if (sequence_count == 0) return;
        
        size_t start_idx = (sequence_head + MAX_SEQUENCE_CAPACITY - sequence_count) % MAX_SEQUENCE_CAPACITY;
        for (size_t i = 0; i < sequence_count; ++i) {
            size_t idx = (start_idx + i) % MAX_SEQUENCE_CAPACITY;
            func(sequence_buffer[idx]);
        }
    }
    
    // Hybrid mode
    std::unique_ptr<KalmanDecoder> kalman;
    float nn_weight = 0.7f;  // Default: 70% NN, 30% Kalman
    
    // Statistics
    LatencyTracker inference_latency;
    LatencyTracker preprocessing_latency;
    LatencyTracker postprocessing_latency;
    LatencyTracker nn_latency;
    LatencyTracker kalman_latency;
    std::atomic<uint64_t> total_inferences{0};
    std::atomic<uint64_t> successful_inferences{0};
    std::atomic<uint64_t> failed_inferences{0};
    std::atomic<uint64_t> timeouts{0};
    size_t peak_memory = 0;
    bool using_gpu = false;
    std::string active_backend;
    
    // Config reference
    NeuralNetDecoder::Config config;
    
    Impl(const NeuralNetDecoder::Config& cfg) : config(cfg) {
        max_sequence_length = cfg.sequence_length;
        
        if (cfg.hybrid_mode) {
            kalman = std::make_unique<KalmanDecoder>(cfg.kalman_config);
        }
        
#ifdef PHANTOMCORE_ENABLE_ONNX
        // Configure session options
        session_options.SetIntraOpNumThreads(cfg.num_threads);
        session_options.SetGraphOptimizationLevel(
            cfg.enable_optimization ? GraphOptimizationLevel::ORT_ENABLE_ALL 
                                    : GraphOptimizationLevel::ORT_DISABLE_ALL
        );
        
        // Add GPU execution provider if requested
        if (cfg.use_gpu) {
#ifdef PHANTOMCORE_ENABLE_CUDA
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = cfg.gpu_device_id >= 0 ? cfg.gpu_device_id : 0;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            using_gpu = true;
            active_backend = "ONNX Runtime (CUDA)";
#else
            active_backend = "ONNX Runtime (CPU)";
#endif
        } else {
            active_backend = "ONNX Runtime (CPU)";
        }
#endif
    }
    
#ifdef PHANTOMCORE_ENABLE_ONNX
    NNError create_session(const void* model_data, size_t model_size) {
        try {
            session = std::make_unique<Ort::Session>(
                env, model_data, model_size, session_options
            );
            
            // Get input/output info
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Input info
            size_t num_inputs = session->GetInputCount();
            for (size_t i = 0; i < num_inputs; ++i) {
                auto name = session->GetInputNameAllocated(i, allocator);
                input_names_storage.push_back(name.get());
                
                auto type_info = session->GetInputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                input_shape = tensor_info.GetShape();
            }
            
            // Output info
            size_t num_outputs = session->GetOutputCount();
            for (size_t i = 0; i < num_outputs; ++i) {
                auto name = session->GetOutputNameAllocated(i, allocator);
                output_names_storage.push_back(name.get());
                
                auto type_info = session->GetOutputTypeInfo(i);
                auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
                output_shape = tensor_info.GetShape();
            }
            
            // Build name pointer arrays
            input_names.clear();
            output_names.clear();
            for (const auto& name : input_names_storage) {
                input_names.push_back(name.c_str());
            }
            for (const auto& name : output_names_storage) {
                output_names.push_back(name.c_str());
            }
            
            // Pre-allocate buffers
            size_t input_size = 1;
            for (auto dim : input_shape) {
                if (dim > 0) input_size *= static_cast<size_t>(dim);
            }
            input_buffer.resize(input_size, 0.0f);
            
            size_t output_size = 1;
            for (auto dim : output_shape) {
                if (dim > 0) output_size *= static_cast<size_t>(dim);
            }
            output_buffer.resize(output_size, 0.0f);
            
            peak_memory = (input_size + output_size) * sizeof(float);
            
            return NNError::Success;
            
        } catch (const Ort::Exception& e) {
            (void)e;
            return NNError::InvalidModel;
        } catch (...) {
            return NNError::InvalidModel;
        }
    }
    
    std::expected<DecoderOutput, NNError> run_inference(std::span<const float> input) {
        if (!session) {
            return std::unexpected(NNError::ModelNotLoaded);
        }
        
        auto total_start = Clock::now();
        total_inferences++;
        
        try {
            // Preprocessing
            auto preprocess_start = Clock::now();
            
            // Copy input data
            size_t copy_size = std::min(input.size(), input_buffer.size());
            std::copy(input.begin(), input.begin() + copy_size, input_buffer.begin());
            
            // Handle dynamic batch dimension
            std::vector<int64_t> actual_input_shape = input_shape;
            for (auto& dim : actual_input_shape) {
                if (dim < 0) dim = 1;  // Replace dynamic dims with 1
            }
            
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                input_buffer.data(),
                input_buffer.size(),
                actual_input_shape.data(),
                actual_input_shape.size()
            );
            
            preprocessing_latency.record(Clock::now() - preprocess_start);
            
            // Run inference
            auto nn_start = Clock::now();
            
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(),
                &input_tensor,
                1,
                output_names.data(),
                output_names.size()
            );
            
            nn_latency.record(Clock::now() - nn_start);
            
            // Postprocessing
            auto postprocess_start = Clock::now();
            
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            
            DecoderOutput result;
            result.position.x = output_data[0];
            result.position.y = output_data[1];
            result.velocity.vx = output_data[2];
            result.velocity.vy = output_data[3];
            result.confidence = 1.0f;
            
            postprocessing_latency.record(Clock::now() - postprocess_start);
            
            result.processing_time = Clock::now() - total_start;
            inference_latency.record(result.processing_time);
            successful_inferences++;
            
            return result;
            
        } catch (const Ort::Exception& e) {
            (void)e;
            failed_inferences++;
            return std::unexpected(NNError::InferenceFailed);
        } catch (...) {
            failed_inferences++;
            return std::unexpected(NNError::InferenceFailed);
        }
    }
#endif
};

NeuralNetDecoder::NeuralNetDecoder(const Config& config)
    : impl_(std::make_unique<Impl>(config))
    , config_(config)
{
}

NeuralNetDecoder::~NeuralNetDecoder() = default;

NeuralNetDecoder::NeuralNetDecoder(NeuralNetDecoder&&) noexcept = default;
NeuralNetDecoder& NeuralNetDecoder::operator=(NeuralNetDecoder&&) noexcept = default;

std::expected<void, NNError> NeuralNetDecoder::load_model(const std::filesystem::path& model_path) {
    if (!std::filesystem::exists(model_path)) {
        return std::unexpected(NNError::FileNotFound);
    }
    
#ifdef PHANTOMCORE_ENABLE_ONNX
    try {
        // Read file into buffer
        std::ifstream file(model_path, std::ios::binary | std::ios::ate);
        if (!file) {
            return std::unexpected(NNError::FileNotFound);
        }
        
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(static_cast<size_t>(size));
        if (!file.read(buffer.data(), size)) {
            return std::unexpected(NNError::InvalidModel);
        }
        
        auto error = impl_->create_session(buffer.data(), buffer.size());
        if (error != NNError::Success) {
            return std::unexpected(error);
        }
        
        impl_->model_loaded = true;
        impl_->model_info.name = model_path.stem().string();
        impl_->model_info.model_size_bytes = static_cast<size_t>(size);
        impl_->model_info.input_channels = config_.channel_config.num_channels;
        
        return {};
        
    } catch (...) {
        return std::unexpected(NNError::InvalidModel);
    }
#else
    (void)model_path;
    return std::unexpected(NNError::BackendNotAvailable);
#endif
}

std::expected<void, NNError> NeuralNetDecoder::load_model(std::span<const uint8_t> model_data) {
#ifdef PHANTOMCORE_ENABLE_ONNX
    auto error = impl_->create_session(model_data.data(), model_data.size());
    if (error != NNError::Success) {
        return std::unexpected(error);
    }
    
    impl_->model_loaded = true;
    impl_->model_info.model_size_bytes = model_data.size();
    impl_->model_info.input_channels = config_.channel_config.num_channels;
    
    return {};
#else
    (void)model_data;
    return std::unexpected(NNError::BackendNotAvailable);
#endif
}

std::expected<void, NNError> NeuralNetDecoder::load_builtin_model(const std::string& model_name) {
    // Built-in models would be embedded in the library or downloaded
    // For now, return not found - user needs to provide ONNX file
    (void)model_name;
    return std::unexpected(NNError::FileNotFound);
}

bool NeuralNetDecoder::is_model_loaded() const {
    return impl_->model_loaded;
}

std::optional<NNModelInfo> NeuralNetDecoder::get_model_info() const {
    if (impl_->model_loaded) {
        return impl_->model_info;
    }
    return std::nullopt;
}

void NeuralNetDecoder::unload_model() {
#ifdef PHANTOMCORE_ENABLE_ONNX
    impl_->session.reset();
    impl_->input_names_storage.clear();
    impl_->output_names_storage.clear();
    impl_->input_names.clear();
    impl_->output_names.clear();
#endif
    impl_->model_loaded = false;
    impl_->clear_sequence();
}

std::expected<DecoderOutput, NNError> NeuralNetDecoder::decode(const SpikeData& spike_data) {
    return decode(std::span<const float>(spike_data.data(), spike_data.size()));
}

std::expected<DecoderOutput, NNError> NeuralNetDecoder::decode(std::span<const float> spike_counts) {
    if (!impl_->model_loaded) {
        return std::unexpected(NNError::ModelNotLoaded);
    }
    
#ifdef PHANTOMCORE_ENABLE_ONNX
    // For recurrent models, update sequence buffer
    if (impl_->model_info.sequence_length > 1) {
        // Ensure sequence buffer is initialized (one-time allocation at startup)
        impl_->init_sequence_buffer(spike_counts.size());
        
        // Push to circular buffer (no allocation)
        impl_->push_to_sequence(spike_counts);
        
        // Flatten sequence for input (reuse pre-allocated input_buffer when possible)
        std::vector<float> sequence_input;
        sequence_input.reserve(impl_->sequence_count * spike_counts.size());
        impl_->for_each_sequence_sample([&](const std::vector<float>& sample) {
            sequence_input.insert(sequence_input.end(), sample.begin(), sample.end());
        });
        
        auto nn_result = impl_->run_inference(sequence_input);
        
        if (!nn_result) {
            return std::unexpected(nn_result.error());
        }
        
        // Hybrid mode: blend with Kalman
        if (config_.hybrid_mode && impl_->kalman) {
            auto kalman_start = Clock::now();
            
            SpikeData spike_copy(spike_counts.size());
            for (size_t i = 0; i < spike_counts.size(); ++i) {
                spike_copy[i] = spike_counts[i];
            }
            auto kalman_result = impl_->kalman->decode(spike_copy);
            
            impl_->kalman_latency.record(Clock::now() - kalman_start);
            
            float nn_w = impl_->nn_weight;
            float kalman_w = 1.0f - nn_w;
            
            DecoderOutput blended;
            blended.position.x = nn_w * nn_result->position.x + kalman_w * kalman_result.position.x;
            blended.position.y = nn_w * nn_result->position.y + kalman_w * kalman_result.position.y;
            blended.velocity.vx = nn_w * nn_result->velocity.vx + kalman_w * kalman_result.velocity.vx;
            blended.velocity.vy = nn_w * nn_result->velocity.vy + kalman_w * kalman_result.velocity.vy;
            blended.confidence = nn_result->confidence;
            blended.processing_time = nn_result->processing_time;
            
            return blended;
        }
        
        return nn_result;
        
    } else {
        // Non-recurrent: direct inference
        auto nn_result = impl_->run_inference(spike_counts);
        
        if (!nn_result) {
            return std::unexpected(nn_result.error());
        }
        
        // Hybrid mode
        if (config_.hybrid_mode && impl_->kalman) {
            auto kalman_start = Clock::now();
            
            SpikeData spike_copy(spike_counts.size());
            for (size_t i = 0; i < spike_counts.size(); ++i) {
                spike_copy[i] = spike_counts[i];
            }
            auto kalman_result = impl_->kalman->decode(spike_copy);
            
            impl_->kalman_latency.record(Clock::now() - kalman_start);
            
            float nn_w = impl_->nn_weight;
            float kalman_w = 1.0f - nn_w;
            
            DecoderOutput blended;
            blended.position.x = nn_w * nn_result->position.x + kalman_w * kalman_result.position.x;
            blended.position.y = nn_w * nn_result->position.y + kalman_w * kalman_result.position.y;
            blended.velocity.vx = nn_w * nn_result->velocity.vx + kalman_w * kalman_result.velocity.vx;
            blended.velocity.vy = nn_w * nn_result->velocity.vy + kalman_w * kalman_result.velocity.vy;
            blended.confidence = nn_result->confidence;
            blended.processing_time = nn_result->processing_time;
            
            return blended;
        }
        
        return nn_result;
    }
#else
    (void)spike_counts;
    return std::unexpected(NNError::BackendNotAvailable);
#endif
}

std::expected<std::vector<DecoderOutput>, NNError> NeuralNetDecoder::decode_batch(
    std::span<const float> spike_batch,
    size_t num_samples
) {
    if (!impl_->model_loaded) {
        return std::unexpected(NNError::ModelNotLoaded);
    }
    
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

std::expected<DecoderOutput, NNError> NeuralNetDecoder::predict() {
    if (!impl_->model_loaded) {
        return std::unexpected(NNError::ModelNotLoaded);
    }
    
    // For recurrent models, run inference on current sequence
    if (impl_->sequence_count > 0) {
        std::vector<float> sequence_input;
        impl_->for_each_sequence_sample([&](const std::vector<float>& sample) {
            sequence_input.insert(sequence_input.end(), sample.begin(), sample.end());
        });
        
#ifdef PHANTOMCORE_ENABLE_ONNX
        return impl_->run_inference(sequence_input);
#endif
    }
    
    return std::unexpected(NNError::InvalidModel);
}

void NeuralNetDecoder::reset() {
    impl_->clear_sequence();
    
    if (impl_->kalman) {
        impl_->kalman->reset();
    }
}

void NeuralNetDecoder::push_sample(const SpikeData& spike_data) {
    impl_->init_sequence_buffer(spike_data.size());
    impl_->push_to_sequence(std::span<const float>(spike_data.data(), spike_data.size()));
}

size_t NeuralNetDecoder::current_sequence_length() const {
    return impl_->sequence_count;
}

void NeuralNetDecoder::clear_sequence() {
    impl_->clear_sequence();
}

void NeuralNetDecoder::set_nn_weight(float nn_weight) {
    impl_->nn_weight = std::clamp(nn_weight, 0.0f, 1.0f);
}

float NeuralNetDecoder::get_nn_weight() const {
    return impl_->nn_weight;
}

KalmanDecoder* NeuralNetDecoder::kalman_decoder() {
    return impl_->kalman.get();
}

const KalmanDecoder* NeuralNetDecoder::kalman_decoder() const {
    return impl_->kalman.get();
}

NeuralNetDecoder::PerformanceStats NeuralNetDecoder::get_stats() const {
    PerformanceStats stats;
    stats.total_inferences = impl_->total_inferences.load();
    stats.successful_inferences = impl_->successful_inferences.load();
    stats.failed_inferences = impl_->failed_inferences.load();
    stats.timeouts = impl_->timeouts.load();
    stats.inference_latency = impl_->inference_latency.get_stats();
    stats.preprocessing_latency = impl_->preprocessing_latency.get_stats();
    stats.postprocessing_latency = impl_->postprocessing_latency.get_stats();
    stats.nn_latency = impl_->nn_latency.get_stats();
    stats.kalman_latency = impl_->kalman_latency.get_stats();
    stats.peak_memory_usage_bytes = impl_->peak_memory;
    stats.using_gpu = impl_->using_gpu;
    stats.active_backend = impl_->active_backend;
    return stats;
}

void NeuralNetDecoder::reset_stats() {
    impl_->total_inferences = 0;
    impl_->successful_inferences = 0;
    impl_->failed_inferences = 0;
    impl_->timeouts = 0;
    impl_->inference_latency = LatencyTracker{};
    impl_->preprocessing_latency = LatencyTracker{};
    impl_->postprocessing_latency = LatencyTracker{};
    impl_->nn_latency = LatencyTracker{};
    impl_->kalman_latency = LatencyTracker{};
}

void NeuralNetDecoder::warmup(size_t num_warmup_runs) {
    if (!impl_->model_loaded) return;
    
    SpikeData dummy(config_.channel_config.num_channels);
    for (size_t i = 0; i < dummy.size(); ++i) {
        dummy[i] = 0.0f;
    }
    
    for (size_t i = 0; i < num_warmup_runs; ++i) {
        (void)decode(dummy);
    }
    
    // Reset stats after warmup
    reset_stats();
}

LatencyStats NeuralNetDecoder::benchmark(size_t num_iterations) {
    if (!impl_->model_loaded) {
        return {};
    }
    
    LatencyTracker tracker;
    SpikeData dummy(config_.channel_config.num_channels);
    
    for (size_t i = 0; i < num_iterations; ++i) {
        auto start = Clock::now();
        (void)decode(dummy);
        tracker.record(Clock::now() - start);
    }
    
    return tracker.get_stats();
}

bool NeuralNetDecoder::is_backend_available(NNBackend backend) {
    switch (backend) {
        case NNBackend::ONNX:
#ifdef PHANTOMCORE_ENABLE_ONNX
            return true;
#else
            return false;
#endif
        case NNBackend::TensorRT:
#if defined(PHANTOMCORE_ENABLE_ONNX) && defined(PHANTOMCORE_ENABLE_TENSORRT)
            return true;
#else
            return false;
#endif
        default:
            return false;
    }
}

std::vector<NNBackend> NeuralNetDecoder::get_available_backends() {
    std::vector<NNBackend> backends;
    
#ifdef PHANTOMCORE_ENABLE_ONNX
    backends.push_back(NNBackend::ONNX);
#endif
#ifdef PHANTOMCORE_ENABLE_TENSORRT
    backends.push_back(NNBackend::TensorRT);
#endif
    
    return backends;
}

NNBackend NeuralNetDecoder::get_recommended_backend() {
#ifdef PHANTOMCORE_ENABLE_TENSORRT
    return NNBackend::TensorRT;
#elif defined(PHANTOMCORE_ENABLE_ONNX)
    return NNBackend::ONNX;
#else
    return NNBackend::ONNX;  // Default even if not available
#endif
}

std::expected<NNModelInfo, NNError> NeuralNetDecoder::validate_model(
    const std::filesystem::path& model_path
) {
    if (!std::filesystem::exists(model_path)) {
        return std::unexpected(NNError::FileNotFound);
    }
    
    NNModelInfo info;
    info.name = model_path.stem().string();
    info.model_size_bytes = static_cast<size_t>(std::filesystem::file_size(model_path));
    
    // Basic validation - could be extended with ONNX parsing
    return info;
}

// ============================================================================
// NNModelBuilder Implementation
// ============================================================================

std::vector<NNModelBuilder::LayerConfig> NNModelBuilder::mlp(std::vector<size_t> hidden_sizes) {
    std::vector<LayerConfig> layers;
    
    for (size_t units : hidden_sizes) {
        layers.push_back({LayerConfig::Type::Dense, units, 0.0f, 0});
        layers.push_back({LayerConfig::Type::ReLU, 0, 0.0f, 0});
    }
    
    // Output layer
    layers.push_back({LayerConfig::Type::Dense, 4, 0.0f, 0});  // 4 = [x, y, vx, vy]
    
    return layers;
}

std::vector<NNModelBuilder::LayerConfig> NNModelBuilder::lstm(
    size_t hidden_size,
    size_t num_layers,
    float dropout
) {
    std::vector<LayerConfig> layers;
    
    for (size_t i = 0; i < num_layers; ++i) {
        layers.push_back({LayerConfig::Type::LSTM, hidden_size, dropout, 0});
    }
    
    layers.push_back({LayerConfig::Type::Dense, 4, 0.0f, 0});
    
    return layers;
}

std::vector<NNModelBuilder::LayerConfig> NNModelBuilder::tcn(
    std::vector<size_t> channel_sizes,
    size_t kernel_size
) {
    std::vector<LayerConfig> layers;
    
    for (size_t channels : channel_sizes) {
        layers.push_back({LayerConfig::Type::Conv1D, channels, 0.0f, kernel_size});
        layers.push_back({LayerConfig::Type::BatchNorm, 0, 0.0f, 0});
        layers.push_back({LayerConfig::Type::ReLU, 0, 0.0f, 0});
    }
    
    layers.push_back({LayerConfig::Type::Dense, 4, 0.0f, 0});
    
    return layers;
}

std::string NNModelBuilder::generate_training_script(
    const std::vector<LayerConfig>& architecture,
    size_t input_channels,
    size_t output_dim,
    const std::string& model_name
) {
    std::ostringstream script;
    
    script << R"(#!/usr/bin/env python3
"""
Auto-generated PyTorch training script for PhantomCore decoder model.
Train with PhantomLink data and export to ONNX for deployment.

Usage:
    python train_model.py --data_path /path/to/training_data.npz
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from pathlib import Path

class DecoderModel(nn.Module):
    """Neural decoder model for BCI applications."""
    
    def __init__(self, input_dim: int = )" << input_channels << R"(, output_dim: int = )" << output_dim << R"():
        super().__init__()
        
        layers = []
)";

    size_t prev_dim = input_channels;
    for (const auto& layer : architecture) {
        switch (layer.type) {
            case LayerConfig::Type::Dense:
                script << "        layers.append(nn.Linear(" << prev_dim << ", " << layer.units << "))\n";
                prev_dim = layer.units;
                break;
            case LayerConfig::Type::ReLU:
                script << "        layers.append(nn.ReLU())\n";
                break;
            case LayerConfig::Type::Tanh:
                script << "        layers.append(nn.Tanh())\n";
                break;
            case LayerConfig::Type::Dropout:
                script << "        layers.append(nn.Dropout(" << layer.dropout << "))\n";
                break;
            case LayerConfig::Type::BatchNorm:
                script << "        layers.append(nn.BatchNorm1d(" << prev_dim << "))\n";
                break;
            default:
                break;
        }
    }
    
    script << R"(        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


def train(model, train_loader, val_loader, epochs=100, lr=1e-3, device='cuda'):
    """Train the model."""
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ')" << model_name << R"(.pt')
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}')
    
    return model


def export_onnx(model, input_dim, output_path, device='cpu'):
    """Export model to ONNX format."""
    model = model.to(device).eval()
    
    dummy_input = torch.randn(1, input_dim).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['spike_counts'],
        output_names=['kinematics'],
        dynamic_axes={
            'spike_counts': {0: 'batch_size'},
            'kinematics': {0: 'batch_size'}
        },
        opset_version=14
    )
    print(f'Model exported to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    # Load data
    data = np.load(args.data_path)
    X = torch.tensor(data['spikes'], dtype=torch.float32)
    y = torch.tensor(data['kinematics'], dtype=torch.float32)
    
    # Split
    n_train = int(0.8 * len(X))
    train_dataset = torch.utils.data.TensorDataset(X[:n_train], y[:n_train])
    val_dataset = torch.utils.data.TensorDataset(X[n_train:], y[n_train:])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Train
    model = DecoderModel(input_dim=X.shape[1])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = train(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr, device=device)
    
    # Export
    model.load_state_dict(torch.load(')" << model_name << R"(.pt'))
    export_onnx(model, X.shape[1], ')" << model_name << R"(.onnx')
)";
    
    return script.str();
}

std::expected<void, NNError> NNModelBuilder::quantize_model(
    const std::filesystem::path& input_path,
    const std::filesystem::path& output_path,
    Quantization quantization
) {
    // Model quantization requires ONNX Runtime quantization tools
    // This would call onnxruntime.quantization in Python or use ORT C++ API
    (void)input_path;
    (void)output_path;
    (void)quantization;
    return std::unexpected(NNError::UnsupportedArchitecture);
}

}  // namespace phantomcore
