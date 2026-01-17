#pragma once

#include "types.hpp"
#include "kalman_decoder.hpp"
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <expected>
#include <functional>
#include <filesystem>

namespace phantomcore {

// ============================================================================
// Neural Network Backend
// ============================================================================

/**
 * @brief Supported neural network inference backends
 */
enum class NNBackend {
    ONNX,           // ONNX Runtime (default, cross-platform)
    TensorRT,       // NVIDIA TensorRT (lowest latency on NVIDIA GPUs)
    CoreML,         // Apple CoreML (best on Apple Silicon)
    OpenVINO,       // Intel OpenVINO (best on Intel CPUs)
    DirectML        // DirectML (Windows GPU acceleration)
};

/**
 * @brief Neural network model architecture types
 */
enum class NNArchitecture {
    MLP,            // Multi-layer perceptron (feedforward)
    LSTM,           // Long Short-Term Memory (recurrent)
    GRU,            // Gated Recurrent Unit (recurrent)
    TCN,            // Temporal Convolutional Network
    Transformer,    // Attention-based transformer
    Hybrid,         // Neural features + Kalman smoothing
    Custom          // User-defined architecture
};

/**
 * @brief Quantization level for model optimization
 */
enum class Quantization {
    None,           // Full FP32 precision
    FP16,           // Half precision (2x memory reduction)
    INT8,           // 8-bit integer (4x memory reduction, fastest)
    INT4            // 4-bit integer (experimental, 8x reduction)
};

/**
 * @brief Error codes for neural network operations
 */
enum class NNError {
    Success = 0,
    ModelNotLoaded,
    InvalidModel,
    InferenceFailed,
    BackendNotAvailable,
    ShapeMismatch,
    OutOfMemory,
    Timeout,
    FileNotFound,
    UnsupportedArchitecture
};

// ============================================================================
// Model Metadata
// ============================================================================

/**
 * @brief Metadata about a neural network model
 */
struct NNModelInfo {
    std::string name;
    std::string version;
    NNArchitecture architecture = NNArchitecture::MLP;
    
    // Input/output shapes
    size_t input_channels = 142;
    size_t sequence_length = 1;         // For recurrent models
    size_t output_dim = 4;              // [x, y, vx, vy]
    
    // Model characteristics
    size_t num_parameters = 0;
    size_t model_size_bytes = 0;
    Quantization quantization = Quantization::None;
    
    // Training info
    std::string trained_on;             // Dataset name
    std::string training_date;
    float validation_r2 = 0.0f;
    
    // Latency estimates (microseconds)
    float estimated_latency_cpu_us = 0.0f;
    float estimated_latency_gpu_us = 0.0f;
};

// ============================================================================
// Neural Network Decoder
// ============================================================================

/**
 * @brief Neural network-based decoder with ONNX Runtime integration
 * 
 * Supports modern deep learning architectures for neural decoding:
 * - **MLP**: Fast feedforward networks (~2-5μs)
 * - **LSTM/GRU**: Recurrent networks for temporal modeling (~5-10μs)
 * - **TCN**: Temporal Convolutional Networks (~3-8μs)
 * - **Transformer**: Attention-based models (~10-20μs)
 * - **Hybrid**: NN feature extraction + Kalman smoothing (~5-8μs)
 * 
 * Architecture (Hybrid mode):
 * ```
 *   Spikes (N) ──┬── [Neural Net] ──▶ Features (K)
 *                │                        │
 *                │                        ▼
 *                │                  [Kalman Filter] ──▶ Smoothed Output
 *                │                        │
 *                └────────────────────────┘
 *                     (residual connection for stability)
 * ```
 * 
 * Performance characteristics:
 * - CPU (AVX2): 5-15μs depending on architecture
 * - GPU (CUDA): 2-8μs depending on architecture
 * - TensorRT: 1-5μs with optimized engines
 * 
 * @note Models must be exported to ONNX format from PyTorch/TensorFlow
 */
class NeuralNetDecoder {
public:
    struct Config {
        /// Channel configuration
        ChannelConfig channel_config = ChannelConfig::mc_maze();
        
        /// Inference backend
        NNBackend backend = NNBackend::ONNX;
        
        /// Use GPU if available
        bool use_gpu = true;
        
        /// GPU device ID (-1 = auto-select)
        int gpu_device_id = -1;
        
        /// Model quantization
        Quantization quantization = Quantization::None;
        
        /// Number of threads for CPU inference
        int num_threads = 4;
        
        /// Enable model optimization (first inference slower, subsequent faster)
        bool enable_optimization = true;
        
        /// Sequence length for recurrent models
        size_t sequence_length = 20;
        
        /// Enable hybrid mode (NN + Kalman)
        bool hybrid_mode = true;
        
        /// Kalman config for hybrid mode
        KalmanDecoder::Config kalman_config;
        
        /// Memory arena size (bytes, 0 = auto)
        size_t memory_arena_size = 0;
        
        /// Inference timeout (microseconds, 0 = no timeout)
        uint64_t inference_timeout_us = 1000;  // 1ms default
        
        Config() = default;
    };
    
    explicit NeuralNetDecoder(const Config& config = Config{});
    ~NeuralNetDecoder();
    
    // Non-copyable, movable
    NeuralNetDecoder(const NeuralNetDecoder&) = delete;
    NeuralNetDecoder& operator=(const NeuralNetDecoder&) = delete;
    NeuralNetDecoder(NeuralNetDecoder&&) noexcept;
    NeuralNetDecoder& operator=(NeuralNetDecoder&&) noexcept;
    
    // ========================================================================
    // Model Loading
    // ========================================================================
    
    /**
     * @brief Load model from ONNX file
     * @param model_path Path to .onnx file
     * @return Success or error
     */
    std::expected<void, NNError> load_model(const std::filesystem::path& model_path);
    
    /**
     * @brief Load model from memory buffer
     * @param model_data ONNX model bytes
     * @return Success or error
     */
    std::expected<void, NNError> load_model(std::span<const uint8_t> model_data);
    
    /**
     * @brief Load pre-built model by name
     * 
     * Available built-in models:
     * - "mlp_small": 2-layer MLP, ~10K params, ~3μs
     * - "mlp_medium": 4-layer MLP, ~100K params, ~5μs
     * - "lstm_64": LSTM with 64 hidden units, ~50K params, ~8μs
     * - "tcn_light": Lightweight TCN, ~30K params, ~4μs
     * - "hybrid_tcn": TCN + Kalman fusion, ~40K params, ~6μs
     */
    std::expected<void, NNError> load_builtin_model(const std::string& model_name);
    
    /**
     * @brief Check if model is loaded
     */
    bool is_model_loaded() const;
    
    /**
     * @brief Get loaded model info
     */
    std::optional<NNModelInfo> get_model_info() const;
    
    /**
     * @brief Unload current model
     */
    void unload_model();
    
    // ========================================================================
    // Decoding
    // ========================================================================
    
    /**
     * @brief Decode spike data to kinematics
     */
    std::expected<DecoderOutput, NNError> decode(const SpikeData& spike_data);
    
    /**
     * @brief Decode with span input
     */
    std::expected<DecoderOutput, NNError> decode(std::span<const float> spike_counts);
    
    /**
     * @brief Batch decode for offline analysis
     * @param spike_batch Matrix [num_samples x num_channels]
     * @param num_samples Number of samples
     * @return Vector of decoded outputs
     */
    std::expected<std::vector<DecoderOutput>, NNError> decode_batch(
        std::span<const float> spike_batch,
        size_t num_samples
    );
    
    /**
     * @brief Predict without new input (for recurrent models)
     */
    std::expected<DecoderOutput, NNError> predict();
    
    /**
     * @brief Reset internal state (for recurrent models)
     */
    void reset();
    
    // ========================================================================
    // Recurrent Model Interface
    // ========================================================================
    
    /**
     * @brief Push sample to sequence buffer (for recurrent models)
     * 
     * Call this to accumulate samples before decoding.
     * Useful when spike data arrives faster than decode rate.
     */
    void push_sample(const SpikeData& spike_data);
    
    /**
     * @brief Get current sequence length
     */
    size_t current_sequence_length() const;
    
    /**
     * @brief Clear sequence buffer
     */
    void clear_sequence();
    
    // ========================================================================
    // Hybrid Mode Control
    // ========================================================================
    
    /**
     * @brief Set blend ratio between NN and Kalman outputs
     * @param nn_weight Weight for NN output (0.0 = pure Kalman, 1.0 = pure NN)
     */
    void set_nn_weight(float nn_weight);
    
    /**
     * @brief Get current NN weight
     */
    float get_nn_weight() const;
    
    /**
     * @brief Access Kalman decoder (for calibration in hybrid mode)
     */
    KalmanDecoder* kalman_decoder();
    const KalmanDecoder* kalman_decoder() const;
    
    // ========================================================================
    // Performance & Diagnostics
    // ========================================================================
    
    struct PerformanceStats {
        uint64_t total_inferences = 0;
        uint64_t successful_inferences = 0;
        uint64_t failed_inferences = 0;
        uint64_t timeouts = 0;
        
        LatencyStats inference_latency;
        LatencyStats preprocessing_latency;
        LatencyStats postprocessing_latency;
        
        // For hybrid mode
        LatencyStats nn_latency;
        LatencyStats kalman_latency;
        
        size_t current_memory_usage_bytes = 0;
        size_t peak_memory_usage_bytes = 0;
        
        bool using_gpu = false;
        std::string active_backend;
    };
    
    PerformanceStats get_stats() const;
    void reset_stats();
    
    /**
     * @brief Warm up model with dummy inference
     * 
     * First inference is typically slower due to lazy initialization.
     * Call this after loading to ensure consistent latency.
     * 
     * @param num_warmup_runs Number of warm-up inferences (default: 10)
     */
    void warmup(size_t num_warmup_runs = 10);
    
    /**
     * @brief Run latency benchmark
     * @param num_iterations Number of iterations
     * @return Latency statistics
     */
    LatencyStats benchmark(size_t num_iterations = 100);
    
    // ========================================================================
    // Static Utilities
    // ========================================================================
    
    /**
     * @brief Check if a backend is available
     */
    static bool is_backend_available(NNBackend backend);
    
    /**
     * @brief Get list of available backends
     */
    static std::vector<NNBackend> get_available_backends();
    
    /**
     * @brief Get recommended backend for current system
     */
    static NNBackend get_recommended_backend();
    
    /**
     * @brief Validate ONNX model file
     */
    static std::expected<NNModelInfo, NNError> validate_model(
        const std::filesystem::path& model_path
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

// ============================================================================
// Model Builder (for creating models programmatically)
// ============================================================================

/**
 * @brief Builder for creating neural network models for decoder
 * 
 * Provides utilities for:
 * - Exporting PyTorch models to ONNX
 * - Creating simple built-in architectures
 * - Quantizing models for deployment
 */
class NNModelBuilder {
public:
    struct LayerConfig {
        enum class Type { Dense, LSTM, GRU, Conv1D, BatchNorm, Dropout, ReLU, Tanh };
        Type type;
        size_t units = 64;
        float dropout = 0.0f;
        size_t kernel_size = 3;  // For Conv1D
    };
    
    /**
     * @brief Create MLP model configuration
     */
    static std::vector<LayerConfig> mlp(
        std::vector<size_t> hidden_sizes = {128, 64, 32}
    );
    
    /**
     * @brief Create LSTM model configuration
     */
    static std::vector<LayerConfig> lstm(
        size_t hidden_size = 64,
        size_t num_layers = 2,
        float dropout = 0.1f
    );
    
    /**
     * @brief Create TCN model configuration
     */
    static std::vector<LayerConfig> tcn(
        std::vector<size_t> channel_sizes = {64, 64, 32},
        size_t kernel_size = 3
    );
    
    /**
     * @brief Export Python training script
     * 
     * Generates a PyTorch training script that can be run in PhantomLink
     * to train a model and export to ONNX.
     */
    static std::string generate_training_script(
        const std::vector<LayerConfig>& architecture,
        size_t input_channels,
        size_t output_dim = 4,
        const std::string& model_name = "decoder_model"
    );
    
    /**
     * @brief Quantize an ONNX model
     * @param input_path Original model path
     * @param output_path Quantized model output path
     * @param quantization Target quantization level
     * @return Success or error
     */
    static std::expected<void, NNError> quantize_model(
        const std::filesystem::path& input_path,
        const std::filesystem::path& output_path,
        Quantization quantization
    );
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Convert NN error to string
 */
const char* nn_error_string(NNError error);

/**
 * @brief Get string name of backend
 */
const char* nn_backend_string(NNBackend backend);

/**
 * @brief Get string name of architecture
 */
const char* nn_architecture_string(NNArchitecture arch);

}  // namespace phantomcore
