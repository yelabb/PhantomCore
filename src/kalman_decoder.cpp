#include "phantomcore/kalman_decoder.hpp"
#include "phantomcore/dimensionality_reduction.hpp"
#include "phantomcore/regularization.hpp"
#include "phantomcore/simd_utils.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <algorithm>

namespace phantomcore {

// ============================================================================
// KalmanDecoder Implementation
// ============================================================================

struct KalmanDecoder::Impl {
    StateVector state;
    StateMatrix covariance;
    KalmanGain kalman_gain;
    
    // Pre-computed matrices for efficiency
    StateMatrix A;           // State transition
    StateMatrix Q;           // Process noise
    ObsMatrix H;             // Observation model (raw space)
    Eigen::Matrix<float, OBS_DIM, OBS_DIM> R;  // Measurement noise
    
    // === Latent Space Decoding ===
    std::unique_ptr<PCAProjector> pca;
    Eigen::MatrixXf H_latent;  // Observation model in latent space [latent_dim x 4]
    Eigen::MatrixXf R_latent;  // Measurement noise in latent space
    bool use_latent = false;
    size_t latent_dim = 0;
    
    // For observation model training
    Eigen::Matrix<float, OBS_DIM, OBS_DIM> H_transpose_R_inv;
    
    // Statistics
    LatencyTracker latency_tracker{1000};
    Duration max_decode_time{};
    uint64_t total_decodes = 0;
    float last_innovation_magnitude = 0.0f;
    
    // Spike normalization
    AlignedSpikeData spike_mean{};
    AlignedSpikeData spike_std{};
    size_t calibration_samples = 0;
    
    // Calibration results
    float last_r2_score = 0.0f;
    float last_cv_score = 0.0f;
    float last_lambda = 0.0f;
    
    Impl() {
        spike_mean.counts.fill(0.0f);
        spike_std.counts.fill(1.0f);
    }
};

KalmanDecoder::KalmanDecoder(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    
    // Initialize state
    impl_->state = config.initial_state;
    impl_->covariance = config.initial_covariance;
    
    // Set up constant velocity model if not specified
    impl_->A = config.state_transition;
    if (impl_->A.isIdentity()) {
        // Default: constant velocity model
        // [x']   [1 0 dt 0 ] [x ]
        // [y'] = [0 1 0  dt] [y ]
        // [vx']  [0 0 1  0 ] [vx]
        // [vy']  [0 0 0  1 ] [vy]
        impl_->A << 1, 0, config.dt, 0,
                    0, 1, 0, config.dt,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
    }
    
    impl_->Q = config.process_noise;
    impl_->H = config.observation_model;
    impl_->R = config.measurement_noise;
}

KalmanDecoder::~KalmanDecoder() = default;
KalmanDecoder::KalmanDecoder(KalmanDecoder&&) noexcept = default;
KalmanDecoder& KalmanDecoder::operator=(KalmanDecoder&&) noexcept = default;

DecoderOutput KalmanDecoder::decode(const SpikeCountArray& spike_counts) {
    // Convert to aligned data
    AlignedSpikeData aligned;
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        aligned[i] = static_cast<float>(spike_counts[i]);
    }
    return decode(aligned);
}

DecoderOutput KalmanDecoder::decode(const AlignedSpikeData& spike_counts) {
    auto start = Clock::now();
    
    // Normalize spikes (z-score)
    AlignedSpikeData normalized;
    simd::ChannelProcessor::compute_zscores(
        spike_counts,
        impl_->spike_mean,
        impl_->spike_std,
        normalized
    );
    
    // === PREDICT ===
    // x_pred = A * x
    StateVector x_pred = impl_->A * impl_->state;
    
    // P_pred = A * P * A^T + Q
    StateMatrix P_pred = impl_->A * impl_->covariance * impl_->A.transpose() + impl_->Q;
    
    // === UPDATE ===
    // Branch based on whether we use latent space or raw observations
    
    if (impl_->use_latent && impl_->pca && impl_->pca->is_fitted()) {
        // =====================================================================
        // LATENT SPACE KALMAN UPDATE (PCA-reduced, much faster)
        // =====================================================================
        // Project to latent space: 142 dims → k dims (typically 15)
        Eigen::Map<const Eigen::VectorXf> z_raw(normalized.data(), NUM_CHANNELS);
        Eigen::VectorXf z_latent = impl_->pca->transform(z_raw);
        
        const Eigen::Index k = static_cast<Eigen::Index>(impl_->latent_dim);
        
        // Innovation in latent space: y = z_latent - H_latent * x_pred
        // Note: H_latent is [k x 4], maps state to latent observations
        Eigen::VectorXf y_latent = z_latent - impl_->H_latent * x_pred;
        impl_->last_innovation_magnitude = y_latent.norm();
        
        // Kalman update with k-dimensional observations (k << 142)
        // S = H_latent * P_pred * H_latent^T + R_latent  [k x k]
        Eigen::MatrixXf S = impl_->H_latent * P_pred * impl_->H_latent.transpose() + impl_->R_latent;
        
        // K = P_pred * H_latent^T * S^-1  [4 x k]
        Eigen::MatrixXf K = P_pred * impl_->H_latent.transpose() * S.inverse();
        
        // Updated state: x = x_pred + K * y_latent
        impl_->state = x_pred + K * y_latent;
        
        // Updated covariance: P = (I - K * H_latent) * P_pred
        StateMatrix I = StateMatrix::Identity();
        impl_->covariance = (I - K * impl_->H_latent) * P_pred;
        
    } else {
        // =====================================================================
        // RAW OBSERVATION KALMAN UPDATE (142 dimensions - Woodbury optimized)
        // =====================================================================
        Eigen::Map<const ObsVector> z(normalized.data());
        
        // Innovation: y = z - H * x_pred
        ObsVector y = z - impl_->H * x_pred;
        impl_->last_innovation_magnitude = y.norm();
        
        // Woodbury identity for efficient update
        auto H_T = impl_->H.transpose();
        float r_inv = 1.0f / impl_->R(0, 0);
        
        StateMatrix H_T_R_inv_H = H_T * (r_inv * impl_->H);
        StateMatrix P_pred_inv = P_pred.inverse();
        StateMatrix M = (P_pred_inv + H_T_R_inv_H).inverse();
        
        impl_->kalman_gain = M * H_T * r_inv;
        impl_->state = x_pred + impl_->kalman_gain * y;
        impl_->covariance = M;
    }
    
    // Build output
    DecoderOutput output;
    output.position.x = impl_->state(0);
    output.position.y = impl_->state(1);
    output.velocity.vx = impl_->state(2);
    output.velocity.vy = impl_->state(3);
    output.confidence = 1.0f / (1.0f + impl_->last_innovation_magnitude);
    output.processing_time = Clock::now() - start;
    
    // Update stats
    impl_->latency_tracker.record(output.processing_time);
    if (output.processing_time > impl_->max_decode_time) {
        impl_->max_decode_time = output.processing_time;
    }
    impl_->total_decodes++;
    
    return output;
}

DecoderOutput KalmanDecoder::predict() {
    auto start = Clock::now();
    
    // Only prediction step, no measurement update
    impl_->state = impl_->A * impl_->state;
    impl_->covariance = impl_->A * impl_->covariance * impl_->A.transpose() + impl_->Q;
    
    DecoderOutput output;
    output.position.x = impl_->state(0);
    output.position.y = impl_->state(1);
    output.velocity.vx = impl_->state(2);
    output.velocity.vy = impl_->state(3);
    output.confidence = 0.5f;  // Lower confidence for prediction-only
    output.processing_time = Clock::now() - start;
    
    return output;
}

void KalmanDecoder::reset() {
    impl_->state = config_.initial_state;
    impl_->covariance = config_.initial_covariance;
    impl_->total_decodes = 0;
    impl_->max_decode_time = Duration{};
    impl_->latency_tracker.reset();
}

KalmanDecoder::StateVector KalmanDecoder::get_state() const {
    return impl_->state;
}

KalmanDecoder::StateMatrix KalmanDecoder::get_covariance() const {
    return impl_->covariance;
}

void KalmanDecoder::calibrate_legacy(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& kinematics
) {
    calibrate(neural_data, kinematics);
}

KalmanDecoder::CalibrationResult KalmanDecoder::calibrate(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& kinematics
) {
    CalibrationResult result;
    result.n_samples = static_cast<size_t>(neural_data.rows());
    
    if (neural_data.rows() < 10 || neural_data.rows() != kinematics.rows()) {
        return result;
    }
    
    // =========================================================================
    // Step 1: Compute normalization parameters (z-score)
    // =========================================================================
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        impl_->spike_mean[ch] = neural_data.col(static_cast<Eigen::Index>(ch)).mean();
        float variance = (neural_data.col(static_cast<Eigen::Index>(ch)).array() - 
                         impl_->spike_mean[ch]).square().mean();
        impl_->spike_std[ch] = std::sqrt(variance);
        if (impl_->spike_std[ch] < 1e-6f) {
            impl_->spike_std[ch] = 1.0f;
        }
    }
    
    // Normalize neural data
    Eigen::MatrixXf normalized = neural_data;
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        normalized.col(static_cast<Eigen::Index>(ch)) = 
            (neural_data.col(static_cast<Eigen::Index>(ch)).array() - impl_->spike_mean[ch]) / 
            impl_->spike_std[ch];
    }
    
    // =========================================================================
    // Step 2: PCA Dimensionality Reduction (142 → latent_dim)
    // =========================================================================
    Eigen::MatrixXf features;
    
    if (config_.use_pca) {
        PCAProjector::Config pca_cfg;
        pca_cfg.n_components = config_.latent_dim;
        pca_cfg.use_variance_threshold = config_.use_variance_threshold;
        pca_cfg.variance_threshold = config_.pca_variance_threshold;
        pca_cfg.center = true;  // Already normalized, but center in PCA too
        
        impl_->pca = std::make_unique<PCAProjector>(pca_cfg);
        features = impl_->pca->fit_transform(normalized);
        
        if (features.cols() == 0) {
            return result;  // PCA failed
        }
        
        impl_->use_latent = true;
        impl_->latent_dim = impl_->pca->n_components();
        result.latent_dim = impl_->latent_dim;
        result.variance_explained = impl_->pca->cumulative_variance_explained();
    } else {
        features = normalized;
        impl_->use_latent = false;
        impl_->latent_dim = NUM_CHANNELS;
        result.latent_dim = NUM_CHANNELS;
        result.variance_explained = 1.0f;
    }
    
    // =========================================================================
    // Step 3: Ridge Regression with Optional Cross-Validation
    // =========================================================================
    RidgeRegression::Config ridge_cfg;
    ridge_cfg.lambda = config_.ridge_lambda;
    ridge_cfg.fit_intercept = true;
    ridge_cfg.normalize = false;  // Already normalized
    
    RidgeRegression ridge(ridge_cfg);
    
    if (config_.auto_tune_lambda) {
        // Cross-validate to find optimal lambda
        std::vector<float> lambdas = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f};
        auto cv_result = ridge.cross_validate(features, kinematics, lambdas, 5);
        
        result.optimal_lambda = cv_result.best_lambda;
        result.cv_score = cv_result.best_score;
        
        // Refit with optimal lambda
        ridge.set_lambda(cv_result.best_lambda);
    } else {
        result.optimal_lambda = config_.ridge_lambda;
    }
    
    if (!ridge.fit(features, kinematics)) {
        return result;
    }
    
    result.r2_score = ridge.score(features, kinematics);
    impl_->last_r2_score = result.r2_score;
    impl_->last_cv_score = result.cv_score;
    impl_->last_lambda = result.optimal_lambda;
    
    // =========================================================================
    // Step 4: Build Observation Model for Kalman Filter
    // =========================================================================
    // The Ridge model learns: kinematics = features * W + b
    // We need the inverse: H such that H * state ≈ observations
    // 
    // For Kalman, we model: z = H * x + noise
    // where z = latent features, x = [x, y, vx, vy]
    //
    // From calibration: x ≈ features * W (Ridge weights)
    // So: features ≈ x * W^-1 (pseudoinverse)
    // Thus: H_latent = pinv(W)^T
    
    Eigen::MatrixXf W = ridge.coefficients();  // [latent_dim x 4]
    
    if (impl_->use_latent) {
        // Store latent observation model
        // H_latent maps state (4) to latent (k): H_latent is [k x 4]
        // We want: z_latent = H_latent * state
        // From regression: state = features * W => features = state * pinv(W)
        // So H_latent^T = pinv(W) => H_latent = pinv(W)^T
        
        // But for Kalman update, we need H such that innovation = z - H*x
        // Using pseudoinverse of W
        impl_->H_latent = W;  // [latent_dim x 4]
        
        // Initialize latent R matrix
        impl_->R_latent = Eigen::MatrixXf::Identity(
            static_cast<Eigen::Index>(impl_->latent_dim),
            static_cast<Eigen::Index>(impl_->latent_dim)
        ) * 0.1f;
    } else {
        // Full observation space - copy to fixed-size H
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(NUM_CHANNELS) && i < W.rows(); ++i) {
            for (Eigen::Index j = 0; j < 4 && j < W.cols(); ++j) {
                impl_->H(i, j) = W(i, j);
            }
        }
    }
    
    impl_->calibration_samples = result.n_samples;
    result.success = true;
    return result;
}

void KalmanDecoder::load_weights(std::span<const float> observation_weights) {
    if (observation_weights.size() != OBS_DIM * STATE_DIM) {
        throw std::runtime_error("Invalid weight size");
    }
    
    for (size_t i = 0; i < OBS_DIM; ++i) {
        for (size_t j = 0; j < STATE_DIM; ++j) {
            impl_->H(i, j) = observation_weights[i * STATE_DIM + j];
        }
    }
}

std::vector<float> KalmanDecoder::save_weights() const {
    std::vector<float> weights(OBS_DIM * STATE_DIM);
    for (size_t i = 0; i < OBS_DIM; ++i) {
        for (size_t j = 0; j < STATE_DIM; ++j) {
            weights[i * STATE_DIM + j] = impl_->H(i, j);
        }
    }
    return weights;
}

KalmanDecoder::Stats KalmanDecoder::get_stats() const {
    Stats stats;
    auto latency = impl_->latency_tracker.get_stats();
    stats.mean_decode_time = Duration(static_cast<int64_t>(latency.mean_us * 1000));
    stats.max_decode_time = impl_->max_decode_time;
    stats.total_decodes = impl_->total_decodes;
    stats.innovation_magnitude = impl_->last_innovation_magnitude;
    return stats;
}

// ============================================================================
// LinearDecoder Implementation
// ============================================================================

LinearDecoder::LinearDecoder(const Config& config)
    : config_(config) {
    running_mean_.counts.fill(0.0f);
    running_std_.counts.fill(1.0f);
}

DecoderOutput LinearDecoder::decode(const SpikeCountArray& spike_counts) {
    AlignedSpikeData aligned;
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        aligned[i] = static_cast<float>(spike_counts[i]);
    }
    return decode(aligned);
}

DecoderOutput LinearDecoder::decode(const AlignedSpikeData& spike_counts) {
    auto start = Clock::now();
    
    AlignedSpikeData normalized = spike_counts;
    
    // Normalize if enabled
    if (config_.normalize_input && sample_count_ > 10) {
        simd::ChannelProcessor::compute_zscores(
            spike_counts, running_mean_, running_std_, normalized
        );
    }
    
    // Update running statistics
    simd::ChannelProcessor::update_statistics(
        spike_counts, running_mean_, running_std_, sample_count_
    );
    sample_count_++;
    
    // Apply linear decoder
    Vec2 pos = simd::ChannelProcessor::apply_decoder(
        normalized,
        config_.weights_x,
        config_.weights_y,
        config_.bias_x,
        config_.bias_y
    );
    
    DecoderOutput output;
    output.position = pos;
    output.velocity = {0.0f, 0.0f};  // Linear decoder doesn't estimate velocity
    output.confidence = 1.0f;
    output.processing_time = Clock::now() - start;
    
    return output;
}

void LinearDecoder::train(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& positions
) {
    // Compute normalization
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        running_mean_[ch] = neural_data.col(ch).mean();
        float variance = (neural_data.col(ch).array() - running_mean_[ch]).square().mean();
        running_std_[ch] = std::sqrt(variance);
        if (running_std_[ch] < 1e-6f) running_std_[ch] = 1.0f;
    }
    
    // Normalize
    Eigen::MatrixXf X(neural_data.rows(), NUM_CHANNELS + 1);
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        X.col(ch) = (neural_data.col(ch).array() - running_mean_[ch]) / running_std_[ch];
    }
    X.col(NUM_CHANNELS).setOnes();  // Bias term
    
    // Least squares: W = (X^T X)^-1 X^T Y
    Eigen::MatrixXf XtX = X.transpose() * X;
    Eigen::MatrixXf XtY = X.transpose() * positions;
    Eigen::MatrixXf W = XtX.ldlt().solve(XtY);  // [143 x 2]
    
    // Extract weights
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        config_.weights_x[ch] = W(ch, 0);
        config_.weights_y[ch] = W(ch, 1);
    }
    config_.bias_x = W(NUM_CHANNELS, 0);
    config_.bias_y = W(NUM_CHANNELS, 1);
    
    sample_count_ = neural_data.rows();
}

void LinearDecoder::reset() {
    config_.weights_x.fill(0.0f);
    config_.weights_y.fill(0.0f);
    config_.bias_x = 0.0f;
    config_.bias_y = 0.0f;
    running_mean_.counts.fill(0.0f);
    running_std_.counts.fill(1.0f);
    sample_count_ = 0;
}

// ============================================================================
// VelocityKalmanDecoder Implementation
// ============================================================================

struct VelocityKalmanDecoder::Impl {
    Eigen::Vector2f state;
    Eigen::Matrix2f covariance;
    Eigen::Matrix<float, NUM_CHANNELS, 2> H;  // Observation model
    
    AlignedSpikeData spike_mean{};
    AlignedSpikeData spike_std{};
    
    Impl() {
        state.setZero();
        covariance.setIdentity();
        H.setZero();
        spike_mean.counts.fill(0.0f);
        spike_std.counts.fill(1.0f);
    }
};

VelocityKalmanDecoder::VelocityKalmanDecoder(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {}

DecoderOutput VelocityKalmanDecoder::decode(const SpikeCountArray& spike_counts) {
    auto start = Clock::now();
    
    // Normalize spikes
    AlignedSpikeData normalized;
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        normalized[i] = (static_cast<float>(spike_counts[i]) - impl_->spike_mean[i]) / 
                        impl_->spike_std[i];
    }
    
    // Predict
    // For velocity, we assume constant velocity: v_pred = v
    Eigen::Vector2f v_pred = impl_->state;
    Eigen::Matrix2f P_pred = impl_->covariance + config_.process_noise;
    
    // Update
    Eigen::Map<const Eigen::Matrix<float, NUM_CHANNELS, 1>> z(normalized.data());
    
    Eigen::Matrix<float, NUM_CHANNELS, 1> z_pred = impl_->H * v_pred;
    Eigen::Matrix<float, NUM_CHANNELS, 1> y = z - z_pred;
    
    Eigen::Matrix<float, NUM_CHANNELS, NUM_CHANNELS> S = 
        impl_->H * P_pred * impl_->H.transpose() + 
        Eigen::Matrix<float, NUM_CHANNELS, NUM_CHANNELS>::Identity() * config_.measurement_noise;
    
    Eigen::Matrix<float, 2, NUM_CHANNELS> K = P_pred * impl_->H.transpose() * S.inverse();
    
    impl_->state = v_pred + K * y;
    impl_->covariance = (Eigen::Matrix2f::Identity() - K * impl_->H) * P_pred;
    
    // Integrate position
    integrated_position_.x += impl_->state(0) * config_.dt;
    integrated_position_.y += impl_->state(1) * config_.dt;
    
    DecoderOutput output;
    output.position = integrated_position_;
    output.velocity = {impl_->state(0), impl_->state(1)};
    output.confidence = 1.0f;
    output.processing_time = Clock::now() - start;
    
    return output;
}

void VelocityKalmanDecoder::reset() {
    impl_->state.setZero();
    impl_->covariance.setIdentity();
    integrated_position_ = {};
}

void VelocityKalmanDecoder::calibrate(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& velocities
) {
    // Compute normalization
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        impl_->spike_mean[ch] = neural_data.col(ch).mean();
        float variance = (neural_data.col(ch).array() - impl_->spike_mean[ch]).square().mean();
        impl_->spike_std[ch] = std::sqrt(variance);
        if (impl_->spike_std[ch] < 1e-6f) impl_->spike_std[ch] = 1.0f;
    }
    
    // Train observation model (simplified)
    Eigen::MatrixXf X = velocities;  // [N x 2]
    Eigen::MatrixXf Z = neural_data;
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        Z.col(ch) = (Z.col(ch).array() - impl_->spike_mean[ch]) / impl_->spike_std[ch];
    }
    
    auto XtX = X.transpose() * X;
    auto XtZ = X.transpose() * Z;
    Eigen::MatrixXf H_temp = XtX.ldlt().solve(XtZ);  // [2 x 142]
    impl_->H = H_temp.transpose();  // [142 x 2]
}

}  // namespace phantomcore
