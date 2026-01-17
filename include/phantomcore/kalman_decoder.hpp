#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <memory>
#include <span>

namespace phantomcore {

/**
 * @brief Kalman Filter-based neural decoder
 * 
 * Implements a standard Kalman filter for decoding cursor kinematics
 * from neural spike counts. Optimized for real-time performance with
 * pre-computed matrix operations.
 * 
 * State vector: [x, y, vx, vy]^T
 * Observation: 142-dimensional spike counts
 */
class KalmanDecoder {
public:
    /// State dimension (position + velocity in 2D)
    static constexpr size_t STATE_DIM = 4;
    
    /// Observation dimension (neural channels)
    static constexpr size_t OBS_DIM = NUM_CHANNELS;
    
    using StateVector = Eigen::Vector<float, STATE_DIM>;
    using StateMatrix = Eigen::Matrix<float, STATE_DIM, STATE_DIM>;
    using ObsVector = Eigen::Vector<float, OBS_DIM>;
    using ObsMatrix = Eigen::Matrix<float, OBS_DIM, STATE_DIM>;
    using KalmanGain = Eigen::Matrix<float, STATE_DIM, OBS_DIM>;
    
    struct Config {
        /// State transition model (A matrix)
        /// Default: constant velocity model
        StateMatrix state_transition = StateMatrix::Identity();
        
        /// Process noise covariance (Q matrix)
        StateMatrix process_noise = StateMatrix::Identity() * 0.01f;
        
        /// Measurement noise covariance (R matrix)
        Eigen::Matrix<float, OBS_DIM, OBS_DIM> measurement_noise = 
            Eigen::Matrix<float, OBS_DIM, OBS_DIM>::Identity() * 0.1f;
        
        /// Observation model (H matrix) - learned from calibration
        ObsMatrix observation_model = ObsMatrix::Zero();
        
        /// Initial state estimate
        StateVector initial_state = StateVector::Zero();
        
        /// Initial error covariance
        StateMatrix initial_covariance = StateMatrix::Identity();
        
        /// Time step (seconds)
        float dt = 1.0f / 40.0f;  // 40 Hz
    };
    
    explicit KalmanDecoder(const Config& config = {});
    ~KalmanDecoder();
    
    // Non-copyable, movable
    KalmanDecoder(const KalmanDecoder&) = delete;
    KalmanDecoder& operator=(const KalmanDecoder&) = delete;
    KalmanDecoder(KalmanDecoder&&) noexcept;
    KalmanDecoder& operator=(KalmanDecoder&&) noexcept;
    
    /**
     * @brief Process neural data and decode kinematics
     * @param spike_counts 142-channel spike counts
     * @return Decoded position and velocity
     */
    DecoderOutput decode(const SpikeCountArray& spike_counts);
    
    /**
     * @brief Process with aligned spike data (faster)
     */
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
     * @param neural_data Matrix of spike counts [num_samples x num_channels]
     * @param kinematics Matrix of ground truth [num_samples x 4]
     */
    void calibrate(
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

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
};

/**
 * @brief Simple linear decoder baseline
 * Fast but less accurate than Kalman filter
 */
class LinearDecoder {
public:
    struct Config {
        std::array<float, NUM_CHANNELS> weights_x{};
        std::array<float, NUM_CHANNELS> weights_y{};
        float bias_x = 0.0f;
        float bias_y = 0.0f;
        bool normalize_input = true;
    };
    
    explicit LinearDecoder(const Config& config = {});
    
    /**
     * @brief Decode position from spike counts
     */
    DecoderOutput decode(const SpikeCountArray& spike_counts);
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

private:
    Config config_;
    AlignedSpikeData running_mean_{};
    AlignedSpikeData running_std_{};
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
