#pragma once

#include "types.hpp"
#include "kalman_decoder.hpp"
#include <Eigen/Dense>
#include <memory>
#include <span>
#include <optional>
#include <functional>

namespace phantomcore {

// ============================================================================
// Drift Detection
// ============================================================================

/**
 * @brief Drift detection methods for neural signal monitoring
 */
enum class DriftDetectionMethod {
    None,                    // No automatic drift detection
    FiringRateKL,           // KL-divergence on firing rate distributions
    CovarianceShift,        // Track covariance matrix changes
    DecodingError,          // Monitor prediction error increase
    HybridMultiMetric       // Combine multiple metrics
};

/**
 * @brief Result of drift detection analysis
 */
struct DriftDetectionResult {
    bool drift_detected = false;
    float drift_score = 0.0f;          // 0.0 = no drift, 1.0 = severe drift
    float firing_rate_divergence = 0.0f;
    float covariance_distance = 0.0f;
    float prediction_error_ratio = 0.0f;
    size_t samples_since_calibration = 0;
    Duration time_since_calibration{};
    
    /// Recommended action
    enum class Action {
        None,               // Continue normally
        IncreaseLearning,   // Increase adaptation rate
        Recalibrate,        // Full recalibration recommended
        Alert               // Alert user, possible electrode issue
    };
    Action recommended_action = Action::None;
};

// ============================================================================
// Adaptive Calibration Configuration
// ============================================================================

/**
 * @brief Configuration for online adaptive learning
 */
struct AdaptiveConfig {
    // === Learning Rate ===
    float learning_rate = 0.01f;           // Base learning rate for RLS
    float min_learning_rate = 0.001f;      // Minimum learning rate
    float max_learning_rate = 0.1f;        // Maximum learning rate
    bool adaptive_learning_rate = true;    // Auto-adjust based on error
    
    // === Forgetting Factor ===
    float forgetting_factor = 0.995f;      // λ: Weight decay for old samples
    float min_forgetting_factor = 0.99f;   // Never forget faster than this
    float max_forgetting_factor = 0.9999f; // Never forget slower than this
    bool adaptive_forgetting = true;       // Auto-adjust based on drift
    
    // === Update Strategy ===
    enum class UpdateStrategy {
        EveryTimestep,      // Update on every decode
        BatchedUpdates,     // Accumulate and update periodically
        ErrorThreshold,     // Update only when error exceeds threshold
        TimeBased           // Update at fixed intervals
    };
    UpdateStrategy strategy = UpdateStrategy::ErrorThreshold;
    
    size_t batch_size = 10;                // For BatchedUpdates
    float error_threshold = 0.1f;          // For ErrorThreshold
    Duration update_interval{std::chrono::milliseconds(100)};  // For TimeBased
    
    // === Drift Detection ===
    DriftDetectionMethod drift_method = DriftDetectionMethod::HybridMultiMetric;
    float drift_threshold = 0.3f;          // Trigger adaptation above this
    float recalibration_threshold = 0.7f;  // Recommend full recalibration
    size_t drift_window_samples = 100;     // Samples for drift estimation
    
    // === Constraints ===
    float max_weight_change_per_update = 0.1f;  // Prevent sudden jumps
    bool preserve_baseline_weights = true;      // Regularize towards initial
    float baseline_regularization = 0.01f;      // Strength of baseline pull
    
    // === Supervision Mode ===
    enum class SupervisionMode {
        FullySupervised,    // Ground truth always available
        SemiSupervised,     // Intermittent ground truth
        SelfSupervised      // No ground truth, use consistency
    };
    SupervisionMode supervision = SupervisionMode::SemiSupervised;
    
    // === Safety ===
    size_t min_samples_for_update = 10;    // Require this many before adapting
    float max_divergence_from_baseline = 2.0f;  // L2 norm limit
    bool enable_rollback = true;           // Allow reverting bad updates
    size_t rollback_window = 50;           // Keep this many checkpoints
};

// ============================================================================
// Adaptive Decoder
// ============================================================================

/**
 * @brief Kalman decoder with online adaptive calibration
 * 
 * Implements Recursive Least Squares (RLS) for online weight updates:
 * 
 * ```
 * For each new observation (z, y):
 *   1. Compute prediction error: e = y - H*z
 *   2. Update gain: K = P*z / (λ + z'*P*z)
 *   3. Update weights: H = H + K*e'
 *   4. Update covariance: P = (P - K*z'*P) / λ
 * ```
 * 
 * Where:
 * - z = neural features (spike counts or latent projections)
 * - y = target kinematics
 * - H = observation matrix (weights)
 * - P = inverse correlation matrix (RLS state)
 * - λ = forgetting factor (0.99-0.999 typical)
 * 
 * Key features:
 * - O(n²) per update via rank-1 matrix update (not O(n³))
 * - Forgetting factor prioritizes recent data
 * - Drift detection triggers increased adaptation
 * - Automatic rollback if performance degrades
 */
class AdaptiveDecoder {
public:
    struct Config {
        /// Base Kalman decoder configuration
        KalmanDecoder::Config kalman_config;
        
        /// Adaptive learning configuration
        AdaptiveConfig adaptive;
    };
    
    explicit AdaptiveDecoder(const Config& config = {});
    ~AdaptiveDecoder();
    
    // Non-copyable, movable
    AdaptiveDecoder(const AdaptiveDecoder&) = delete;
    AdaptiveDecoder& operator=(const AdaptiveDecoder&) = delete;
    AdaptiveDecoder(AdaptiveDecoder&&) noexcept;
    AdaptiveDecoder& operator=(AdaptiveDecoder&&) noexcept;
    
    // ========================================================================
    // Decoding Interface
    // ========================================================================
    
    /**
     * @brief Decode spike data (same as KalmanDecoder)
     */
    DecoderOutput decode(const SpikeData& spike_data);
    DecoderOutput decode(std::span<const float> spike_counts);
    
    /**
     * @brief Decode and update with ground truth feedback
     * 
     * This is the primary interface for closed-loop adaptation.
     * 
     * @param spike_data Neural data
     * @param ground_truth Actual kinematics (from task or calibration)
     * @return Decoded output (before update applied)
     */
    DecoderOutput decode_and_update(
        const SpikeData& spike_data,
        const Vec4& ground_truth
    );
    
    /**
     * @brief Provide delayed ground truth for previous decode
     * 
     * For cases where ground truth arrives after decoding.
     * Uses internal buffer to match with corresponding neural data.
     * 
     * @param ground_truth Actual kinematics
     * @param timestamp When this ground truth was observed
     */
    void provide_ground_truth(const Vec4& ground_truth, Timestamp timestamp);
    
    /**
     * @brief Predict without update
     */
    DecoderOutput predict();
    
    // ========================================================================
    // Adaptation Control
    // ========================================================================
    
    /**
     * @brief Enable/disable online adaptation
     */
    void set_adaptation_enabled(bool enabled);
    bool is_adaptation_enabled() const;
    
    /**
     * @brief Manually set learning rate
     */
    void set_learning_rate(float rate);
    float get_learning_rate() const;
    
    /**
     * @brief Set forgetting factor
     */
    void set_forgetting_factor(float lambda);
    float get_forgetting_factor() const;
    
    /**
     * @brief Freeze adaptation (keep current weights)
     */
    void freeze();
    
    /**
     * @brief Unfreeze adaptation
     */
    void unfreeze();
    
    /**
     * @brief Force an adaptation step with provided data
     */
    void force_update(
        const Eigen::MatrixXf& neural_batch,
        const Eigen::MatrixXf& kinematics_batch
    );
    
    // ========================================================================
    // Drift Detection
    // ========================================================================
    
    /**
     * @brief Get current drift detection status
     */
    DriftDetectionResult get_drift_status() const;
    
    /**
     * @brief Manually trigger drift analysis
     */
    DriftDetectionResult analyze_drift();
    
    /**
     * @brief Set callback for drift events
     */
    using DriftCallback = std::function<void(const DriftDetectionResult&)>;
    void on_drift_detected(DriftCallback callback);
    
    /**
     * @brief Reset drift baseline (after intentional change)
     */
    void reset_drift_baseline();
    
    // ========================================================================
    // Rollback & Recovery
    // ========================================================================
    
    /**
     * @brief Rollback to previous weights
     * @param steps Number of update steps to undo (0 = all the way to baseline)
     * @return True if rollback successful
     */
    bool rollback(size_t steps = 1);
    
    /**
     * @brief Rollback to initial calibration weights
     */
    void rollback_to_baseline();
    
    /**
     * @brief Get number of available rollback steps
     */
    size_t available_rollback_steps() const;
    
    /**
     * @brief Save current state as new baseline
     */
    void commit_as_baseline();
    
    // ========================================================================
    // Calibration
    // ========================================================================
    
    /**
     * @brief Initial batch calibration (sets baseline)
     */
    KalmanDecoder::CalibrationResult calibrate(
        const Eigen::MatrixXf& neural_data,
        const Eigen::MatrixXf& kinematics
    );
    
    /**
     * @brief Reset to initial state
     */
    void reset();
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    struct AdaptationStats {
        uint64_t total_decodes = 0;
        uint64_t supervised_updates = 0;
        uint64_t unsupervised_updates = 0;
        uint64_t skipped_updates = 0;      // Below error threshold
        uint64_t rollbacks_performed = 0;
        
        float cumulative_prediction_error = 0.0f;
        float recent_prediction_error = 0.0f;  // Last window
        float baseline_prediction_error = 0.0f;
        
        float weight_drift_from_baseline = 0.0f;  // L2 norm
        float learning_rate_current = 0.0f;
        float forgetting_factor_current = 0.0f;
        
        LatencyStats update_latency;
    };
    
    AdaptationStats get_stats() const;
    void reset_stats();
    
    // ========================================================================
    // Access to Underlying Decoder
    // ========================================================================
    
    /**
     * @brief Get read-only access to base Kalman decoder
     */
    const KalmanDecoder& base_decoder() const;
    
    /**
     * @brief Get current state estimate
     */
    KalmanDecoder::StateVector get_state() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Firing Rate Monitor (for drift detection)
// ============================================================================

/**
 * @brief Monitors per-channel firing rate statistics for drift detection
 */
class FiringRateMonitor {
public:
    struct Config {
        size_t num_channels = 142;
        size_t window_size = 100;           // Samples for running stats
        size_t baseline_window = 1000;      // Samples for baseline
        float kl_divergence_threshold = 0.5f;
        
        Config() = default;
    };
    
    explicit FiringRateMonitor(const Config& config = Config{});
    ~FiringRateMonitor();
    
    /**
     * @brief Add new observation
     */
    void observe(std::span<const float> spike_counts);
    
    /**
     * @brief Compute KL divergence from baseline
     */
    float compute_kl_divergence() const;
    
    /**
     * @brief Get per-channel firing rate changes
     */
    std::vector<float> get_rate_changes() const;
    
    /**
     * @brief Check if any channels have significantly changed
     */
    std::vector<size_t> get_drifted_channels(float threshold = 2.0f) const;
    
    /**
     * @brief Reset baseline to current statistics
     */
    void reset_baseline();
    
    /**
     * @brief Check if baseline is established
     */
    bool has_baseline() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ============================================================================
// Supervised Data Buffer
// ============================================================================

/**
 * @brief Circular buffer for matching neural data with delayed ground truth
 */
class SupervisedBuffer {
public:
    struct Sample {
        SpikeData neural_data;
        Timestamp timestamp;
        std::optional<Vec4> ground_truth;
        bool used_for_update = false;
    };
    
    explicit SupervisedBuffer(size_t capacity = 1000);
    ~SupervisedBuffer();
    
    /**
     * @brief Add neural sample
     */
    void push_neural(const SpikeData& data, Timestamp ts);
    
    /**
     * @brief Match ground truth with neural sample by timestamp
     * @return True if match found and assigned
     */
    bool match_ground_truth(const Vec4& ground_truth, Timestamp ts, Duration tolerance);
    
    /**
     * @brief Get samples ready for supervised update
     */
    std::vector<std::pair<SpikeData, Vec4>> get_matched_samples(size_t max_count = 0);
    
    /**
     * @brief Get number of unmatched neural samples
     */
    size_t unmatched_count() const;
    
    /**
     * @brief Clear buffer
     */
    void clear();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace phantomcore
