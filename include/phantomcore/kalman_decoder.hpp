#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <memory>
#include <span>

namespace phantomcore {

// Forward declarations
class PCAProjector;
class RidgeRegression;

/**
 * @brief Kalman Filter-based neural decoder with dimensionality reduction
 * 
 * Production-grade implementation with:
 * 
 * 1. **PCA Dimensionality Reduction**: N channels → K latent dimensions
 *    Reduces noise, computational cost, and improves generalization
 * 
 * 2. **Regularized Calibration**: Ridge regression to prevent overfitting
 *    on short calibration sessions
 * 
 * 3. **Efficient Kalman Update**: Woodbury identity for O(k³) instead of O(n³)
 *    where k = latent dim, n = channels
 * 
 * Pipeline:
 *   Spikes (N) → [PCA] → Latent (K) → [Kalman Filter] → Kinematics (4)
 * 
 * State vector: [x, y, vx, vy]^T
 * 
 * Supports runtime-configurable channel count for different hardware.
 */
class KalmanDecoder {
public:
    /// State dimension (position + velocity in 2D)
    static constexpr size_t STATE_DIM = 4;
    
    /// Default latent dimension after PCA
    static constexpr size_t DEFAULT_LATENT_DIM = 15;
    
    // Dynamic types for runtime channel count
    using StateVector = Eigen::Vector<float, STATE_DIM>;
    using StateMatrix = Eigen::Matrix<float, STATE_DIM, STATE_DIM>;
    using ObsVector = Eigen::VectorXf;       // Dynamic: N channels
    using LatentVector = Eigen::VectorXf;    // Dynamic: K latent dims
    using ObsMatrix = Eigen::MatrixXf;       // Dynamic: N x STATE_DIM
    using KalmanGain = Eigen::MatrixXf;      // Dynamic: STATE_DIM x N or STATE_DIM x K
    
    struct Config {
        /// Channel configuration (runtime)
        ChannelConfig channel_config = ChannelConfig::mc_maze();
        
        /// State transition model (A matrix)
        /// Default: constant velocity model
        StateMatrix state_transition = StateMatrix::Identity();
        
        /// Process noise covariance (Q matrix)
        StateMatrix process_noise = StateMatrix::Identity() * 0.01f;
        
        /// Measurement noise scale (for dynamic R matrix)
        float measurement_noise_scale = 0.1f;
        
        /// Initial state estimate
        StateVector initial_state = StateVector::Zero();
        
        /// Initial error covariance
        StateMatrix initial_covariance = StateMatrix::Identity();
        
        /// Time step (seconds)
        float dt = 1.0f / 40.0f;  // 40 Hz
        
        // === Dimensionality Reduction ===
        bool use_pca = true;                  // Enable PCA projection
        size_t latent_dim = DEFAULT_LATENT_DIM;
        float pca_variance_threshold = 0.95f; // Or use variance explained
        bool use_variance_threshold = false;
        
        // === Regularization ===
        float ridge_lambda = 1.0f;            // Ridge regularization strength
        bool auto_tune_lambda = true;         // Cross-validate to find optimal λ
    };
    
    explicit KalmanDecoder(const Config& config = {});
    
    /// Convenience constructor with channel config
    explicit KalmanDecoder(const ChannelConfig& channel_config);
    
    ~KalmanDecoder();
    
    // Non-copyable, movable
    KalmanDecoder(const KalmanDecoder&) = delete;
    KalmanDecoder& operator=(const KalmanDecoder&) = delete;
    KalmanDecoder(KalmanDecoder&&) noexcept;
    KalmanDecoder& operator=(KalmanDecoder&&) noexcept;
    
    /**
     * @brief Process neural data and decode kinematics
     * @param spike_data Dynamic spike counts
     * @return Decoded position and velocity
     */
    DecoderOutput decode(const SpikeData& spike_data);
    
    /**
     * @brief Process with span (for compatibility)
     */
    DecoderOutput decode(std::span<const float> spike_counts);
    
    /**
     * @brief Legacy: Process with fixed-size array (deprecated)
     */
    [[deprecated("Use decode(SpikeData) or decode(span) for dynamic channels")]]
    DecoderOutput decode(const SpikeCountArray& spike_counts);
    
    /**
     * @brief Legacy: Process with aligned spike data (deprecated)
     */
    [[deprecated("Use decode(SpikeData) for dynamic channels")]]
    DecoderOutput decode(const AlignedSpikeData& spike_counts);
    
    /**
     * @brief Predict next state without measurement update
     * Useful for handling dropped packets
     */
    DecoderOutput predict();
    
    /**
     * @brief Reset filter to initial state
     */
    void reset();
    
    /**
     * @brief Get current state estimate
     */
    StateVector get_state() const;
    
    /**
     * @brief Get current error covariance
     */
    StateMatrix get_covariance() const;
    
    /**
     * @brief Update observation model from calibration data
     * 
     * Performs:
     * 1. Fit PCA on neural data (if enabled)
     * 2. Project to latent space
     * 3. Fit Ridge regression with optional cross-validation
     * 4. Initialize Kalman observation model
     * 
     * @param neural_data Matrix of spike counts [num_samples x num_channels]
     * @param kinematics Matrix of ground truth [num_samples x 4]
     * @return Calibration results including R² score and optimal lambda
     */
    struct CalibrationResult {
        bool success = false;
        float r2_score = 0.0f;          // Training R² 
        float cv_score = 0.0f;          // Cross-validation R² (if auto_tune)
        float optimal_lambda = 0.0f;    // Chosen regularization
        size_t latent_dim = 0;          // Actual latent dimensions used
        float variance_explained = 0.0f; // PCA cumulative variance
        size_t n_samples = 0;
    };
    
    CalibrationResult calibrate(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& kinematics
    );
    
    /**
     * @brief Legacy calibration (no return value)
     */
    void calibrate_legacy(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& kinematics
    );
    
    /**
     * @brief Load pre-trained model weights
     */
    void load_weights(std::span<const float> observation_weights);
    
    /**
     * @brief Save current model weights
     */
    std::vector<float> save_weights() const;
    
    /**
     * @brief Get decoder statistics
     */
    struct Stats {
        Duration mean_decode_time{};
        Duration max_decode_time{};
        uint64_t total_decodes = 0;
        float innovation_magnitude = 0.0f;  // Prediction error magnitude
    };
    Stats get_stats() const;
    
    /**
     * @brief Get channel configuration
     */
    const ChannelConfig& channel_config() const;
    
    /**
     * @brief Get number of channels
     */
    size_t num_channels() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

/**
 * @brief Simple linear decoder baseline
 * Fast but less accurate than Kalman filter
 * 
 * Supports runtime-configurable channel count.
 */
class LinearDecoder {
public:
    struct Config {
        ChannelConfig channel_config = ChannelConfig::mc_maze();
        std::vector<float> weights_x;
        std::vector<float> weights_y;
        float bias_x = 0.0f;
        float bias_y = 0.0f;
        bool normalize_input = true;
        
        Config() = default;
        explicit Config(const ChannelConfig& ch_config) 
            : channel_config(ch_config)
            , weights_x(ch_config.num_channels, 0.0f)
            , weights_y(ch_config.num_channels, 0.0f) {}
    };
    
    explicit LinearDecoder(const Config& config = {});
    explicit LinearDecoder(const ChannelConfig& channel_config);
    
    /**
     * @brief Decode position from spike counts (dynamic)
     */
    DecoderOutput decode(const SpikeData& spike_data);
    DecoderOutput decode(std::span<const float> spike_counts);
    
    /**
     * @brief Legacy: Decode from fixed-size arrays
     */
    [[deprecated("Use decode(SpikeData) for dynamic channels")]]
    DecoderOutput decode(const SpikeCountArray& spike_counts);
    [[deprecated("Use decode(SpikeData) for dynamic channels")]]
    DecoderOutput decode(const AlignedSpikeData& spike_counts);
    
    /**
     * @brief Train decoder using least squares
     */
    void train(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& positions
    );
    
    /**
     * @brief Reset to untrained state
     */
    void reset();
    
    size_t num_channels() const { return config_.channel_config.num_channels; }

private:
    Config config_;
    SpikeData running_mean_;
    SpikeData running_std_;
    size_t sample_count_ = 0;
};

/**
 * @brief Velocity Kalman Filter (VKF) decoder
 * Decodes velocity directly, integrates for position
 */
class VelocityKalmanDecoder {
public:
    static constexpr size_t STATE_DIM = 2;  // [vx, vy]
    
    using StateVector = Eigen::Vector2f;
    using StateMatrix = Eigen::Matrix2f;
    
    struct Config {
        StateMatrix process_noise = StateMatrix::Identity() * 0.1f;
        float measurement_noise = 0.5f;
        float dt = 1.0f / 40.0f;
    };
    
    explicit VelocityKalmanDecoder(const Config& config = {});
    
    DecoderOutput decode(const SpikeCountArray& spike_counts);
    void reset();
    void calibrate(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& velocities
    );

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    Vec2 integrated_position_{};
};

}  // namespace phantomcore
