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
    size_t num_channels = 0;
    
    StateVector state;
    StateMatrix covariance;
    
    // Pre-computed matrices for efficiency (dynamic sizes)
    StateMatrix A;           // State transition [4x4]
    StateMatrix Q;           // Process noise [4x4]
    Eigen::MatrixXf H;       // Observation model [num_channels x 4]
    Eigen::MatrixXf R;       // Measurement noise [num_channels x num_channels]
    float measurement_noise_scale = 0.1f;
    
    // === Latent Space Decoding ===
    std::unique_ptr<PCAProjector> pca;
    Eigen::MatrixXf H_latent;  // Observation model in latent space [latent_dim x 4]
    Eigen::MatrixXf R_latent;  // Measurement noise in latent space
    bool use_latent = false;
    size_t latent_dim = 0;
    
    // Statistics
    LatencyTracker latency_tracker{1000};
    Duration max_decode_time{};
    uint64_t total_decodes = 0;
    float last_innovation_magnitude = 0.0f;
    
    // Spike normalization (dynamic)
    SpikeData spike_mean;
    SpikeData spike_std;
    size_t calibration_samples = 0;
    
    // Calibration results
    float last_r2_score = 0.0f;
    float last_cv_score = 0.0f;
    float last_lambda = 0.0f;
    
    explicit Impl(size_t channels) 
        : num_channels(channels)
        , spike_mean(channels)
        , spike_std(channels)
    {
        for (size_t i = 0; i < channels; ++i) {
            spike_std[i] = 1.0f;
        }
        
        // Initialize dynamic matrices
        H = Eigen::MatrixXf::Zero(channels, STATE_DIM);
        R = Eigen::MatrixXf::Identity(channels, channels) * 0.1f;
    }
};

KalmanDecoder::KalmanDecoder(const Config& config)
    : impl_(std::make_unique<Impl>(config.channel_config.num_channels))
    , config_(config) 
{
    // Initialize state
    impl_->state = config.initial_state;
    impl_->covariance = config.initial_covariance;
    impl_->measurement_noise_scale = config.measurement_noise_scale;
    
    // Set up constant velocity model if not specified
    impl_->A = config.state_transition;
    if (impl_->A.isIdentity()) {
        // Default: constant velocity model
        impl_->A << 1, 0, config.dt, 0,
                    0, 1, 0, config.dt,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
    }
    
    impl_->Q = config.process_noise;
}

KalmanDecoder::KalmanDecoder(const ChannelConfig& channel_config) {
    Config cfg;
    cfg.channel_config = channel_config;
    *this = KalmanDecoder(cfg);
}

KalmanDecoder::KalmanDecoder() {
    Config cfg;
    *this = KalmanDecoder(cfg);
}

KalmanDecoder::~KalmanDecoder() = default;
KalmanDecoder::KalmanDecoder(KalmanDecoder&&) noexcept = default;
KalmanDecoder& KalmanDecoder::operator=(KalmanDecoder&&) noexcept = default;

// Primary decode method using SpikeData
DecoderOutput KalmanDecoder::decode(const SpikeData& spike_data) {
    return decode(spike_data.span());
}

// Span-based decode (main implementation)
DecoderOutput KalmanDecoder::decode(std::span<const float> spike_counts) {
    auto start = Clock::now();
    
    const size_t n = std::min(spike_counts.size(), impl_->num_channels);
    
    // Normalize spikes (z-score)
    SpikeData normalized(n);
    for (size_t i = 0; i < n; ++i) {
        float std = impl_->spike_std[i];
        if (std < 1e-6f) std = 1.0f;
        normalized[i] = (spike_counts[i] - impl_->spike_mean[i]) / std;
    }
    
    // === PREDICT ===
    StateVector x_pred = impl_->A * impl_->state;
    StateMatrix P_pred = impl_->A * impl_->covariance * impl_->A.transpose() + impl_->Q;
    
    // === UPDATE ===
    if (impl_->use_latent && impl_->pca && impl_->pca->is_fitted()) {
        // =====================================================================
        // LATENT SPACE KALMAN UPDATE (PCA-reduced, much faster)
        // =====================================================================
        Eigen::Map<const Eigen::VectorXf> z_map(normalized.data(), n);
        Eigen::VectorXf z_raw = z_map;  // Explicit copy to resolve overload
        Eigen::VectorXf z_latent = impl_->pca->transform(z_raw);
        
        // Innovation in latent space
        Eigen::VectorXf y_latent = z_latent - impl_->H_latent * x_pred;
        impl_->last_innovation_magnitude = y_latent.norm();
        
        // Kalman update with k-dimensional observations
        Eigen::MatrixXf S = impl_->H_latent * P_pred * impl_->H_latent.transpose() + impl_->R_latent;
        Eigen::MatrixXf K = P_pred * impl_->H_latent.transpose() * S.inverse();
        
        impl_->state = x_pred + K * y_latent;
        impl_->covariance = (StateMatrix::Identity() - K * impl_->H_latent) * P_pred;
        
    } else {
        // =====================================================================
        // RAW OBSERVATION KALMAN UPDATE (Woodbury optimized)
        // =====================================================================
        Eigen::Map<const Eigen::VectorXf> z(normalized.data(), n);
        
        // Innovation
        Eigen::VectorXf y = z - impl_->H * impl_->state;
        impl_->last_innovation_magnitude = y.norm();
        
        // Woodbury identity for efficient update
        auto H_T = impl_->H.transpose();
        float r_inv = 1.0f / impl_->measurement_noise_scale;
        
        StateMatrix H_T_R_inv_H = H_T * (r_inv * impl_->H);
        StateMatrix M = (P_pred.inverse() + H_T_R_inv_H).inverse();
        
        Eigen::MatrixXf K = M * H_T * r_inv;
        impl_->state = x_pred + K * y;
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

// Legacy decode methods (deprecated)
DecoderOutput KalmanDecoder::decode(const SpikeCountArray& spike_counts) {
    const size_t n = std::min(spike_counts.size(), impl_->num_channels);
    SpikeData data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(spike_counts[i]);
    }
    return decode(data);
}

DecoderOutput KalmanDecoder::decode(const AlignedSpikeData& spike_counts) {
    const size_t n = std::min(spike_counts.size(), impl_->num_channels);
    return decode(std::span<const float>(spike_counts.data(), n));
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
    
    const size_t n_channels = impl_->num_channels;
    
    if (neural_data.rows() < 10 || neural_data.rows() != kinematics.rows()) {
        return result;
    }
    
    // =========================================================================
    // Step 1: Compute normalization parameters (z-score)
    // =========================================================================
    const size_t cols_to_use = std::min(static_cast<size_t>(neural_data.cols()), n_channels);
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        impl_->spike_mean[ch] = neural_data.col(static_cast<Eigen::Index>(ch)).mean();
        float variance = (neural_data.col(static_cast<Eigen::Index>(ch)).array() - 
                         impl_->spike_mean[ch]).square().mean();
        impl_->spike_std[ch] = std::sqrt(variance);
        if (impl_->spike_std[ch] < 1e-6f) {
            impl_->spike_std[ch] = 1.0f;
        }
    }
    
    // Normalize neural data
    Eigen::MatrixXf normalized = neural_data.leftCols(cols_to_use);
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        normalized.col(static_cast<Eigen::Index>(ch)) = 
            (neural_data.col(static_cast<Eigen::Index>(ch)).array() - impl_->spike_mean[ch]) / 
            impl_->spike_std[ch];
    }
    
    // =========================================================================
    // Step 2: PCA Dimensionality Reduction (N channels → latent_dim)
    // =========================================================================
    Eigen::MatrixXf features;
    
    if (config_.use_pca) {
        PCAProjector::Config pca_cfg;
        pca_cfg.n_components = config_.latent_dim;
        pca_cfg.use_variance_threshold = config_.use_variance_threshold;
        pca_cfg.variance_threshold = config_.pca_variance_threshold;
        pca_cfg.center = true;
        
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
        impl_->latent_dim = n_channels;
        result.latent_dim = n_channels;
        result.variance_explained = 1.0f;
    }
    
    // =========================================================================
    // Step 3: Ridge Regression with Optional Cross-Validation
    // =========================================================================
    RidgeRegression::Config ridge_cfg;
    ridge_cfg.lambda = config_.ridge_lambda;
    ridge_cfg.fit_intercept = true;
    ridge_cfg.normalize = false;
    
    RidgeRegression ridge(ridge_cfg);
    
    if (config_.auto_tune_lambda) {
        std::vector<float> lambdas = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f, 1000.0f};
        auto cv_result = ridge.cross_validate(features, kinematics, lambdas, 5);
        
        result.optimal_lambda = cv_result.best_lambda;
        result.cv_score = cv_result.best_score;
        
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
    Eigen::MatrixXf W = ridge.coefficients();
    
    if (impl_->use_latent) {
        impl_->H_latent = W;  // [latent_dim x 4]
        impl_->R_latent = Eigen::MatrixXf::Identity(
            static_cast<Eigen::Index>(impl_->latent_dim),
            static_cast<Eigen::Index>(impl_->latent_dim)
        ) * 0.1f;
    } else {
        // Dynamic H matrix
        impl_->H = Eigen::MatrixXf::Zero(n_channels, STATE_DIM);
        for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(cols_to_use) && i < W.rows(); ++i) {
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
    const size_t expected = impl_->num_channels * STATE_DIM;
    if (observation_weights.size() != expected) {
        throw std::runtime_error("Invalid weight size: expected " + 
            std::to_string(expected) + ", got " + std::to_string(observation_weights.size()));
    }
    
    for (size_t i = 0; i < impl_->num_channels; ++i) {
        for (size_t j = 0; j < STATE_DIM; ++j) {
            impl_->H(i, j) = observation_weights[i * STATE_DIM + j];
        }
    }
}

std::vector<float> KalmanDecoder::save_weights() const {
    std::vector<float> weights(impl_->num_channels * STATE_DIM);
    for (size_t i = 0; i < impl_->num_channels; ++i) {
        for (size_t j = 0; j < STATE_DIM; ++j) {
            weights[i * STATE_DIM + j] = impl_->H(i, j);
        }
    }
    return weights;
}

// ============================================================================
// Full State Serialization (for ModelCheckpoint)
// ============================================================================

std::pair<std::vector<float>, std::vector<float>> KalmanDecoder::get_normalization_params() const {
    std::vector<float> mean(impl_->num_channels);
    std::vector<float> std(impl_->num_channels);
    
    for (size_t i = 0; i < impl_->num_channels; ++i) {
        mean[i] = impl_->spike_mean[i];
        std[i] = impl_->spike_std[i];
    }
    
    return {mean, std};
}

void KalmanDecoder::set_normalization_params(std::span<const float> mean, std::span<const float> std) {
    if (mean.size() != impl_->num_channels || std.size() != impl_->num_channels) {
        throw std::runtime_error("Normalization params size mismatch");
    }
    
    for (size_t i = 0; i < impl_->num_channels; ++i) {
        impl_->spike_mean[i] = mean[i];
        impl_->spike_std[i] = std[i];
    }
}

std::vector<float> KalmanDecoder::get_observation_matrix() const {
    std::vector<float> H(impl_->H.rows() * impl_->H.cols());
    for (Eigen::Index i = 0; i < impl_->H.rows(); ++i) {
        for (Eigen::Index j = 0; j < impl_->H.cols(); ++j) {
            H[i * impl_->H.cols() + j] = impl_->H(i, j);
        }
    }
    return H;
}

std::vector<float> KalmanDecoder::get_latent_observation_matrix() const {
    if (!impl_->use_latent) return {};
    
    std::vector<float> H_latent(impl_->H_latent.rows() * impl_->H_latent.cols());
    for (Eigen::Index i = 0; i < impl_->H_latent.rows(); ++i) {
        for (Eigen::Index j = 0; j < impl_->H_latent.cols(); ++j) {
            H_latent[i * impl_->H_latent.cols() + j] = impl_->H_latent(i, j);
        }
    }
    return H_latent;
}

void KalmanDecoder::set_observation_matrix(std::span<const float> H, size_t rows, size_t cols) {
    if (H.size() != rows * cols) {
        throw std::runtime_error("Observation matrix size mismatch");
    }
    
    impl_->H.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            impl_->H(i, j) = H[i * cols + j];
        }
    }
}

void KalmanDecoder::set_latent_observation_matrix(std::span<const float> H_latent, size_t rows, size_t cols) {
    if (H_latent.empty()) {
        impl_->use_latent = false;
        return;
    }
    
    if (H_latent.size() != rows * cols) {
        throw std::runtime_error("Latent observation matrix size mismatch");
    }
    
    impl_->H_latent.resize(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            impl_->H_latent(i, j) = H_latent[i * cols + j];
        }
    }
    impl_->latent_dim = rows;
    impl_->use_latent = true;
}

std::array<float, 16> KalmanDecoder::get_state_transition() const {
    std::array<float, 16> A;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            A[i * 4 + j] = impl_->A(i, j);
        }
    }
    return A;
}

std::array<float, 16> KalmanDecoder::get_process_noise() const {
    std::array<float, 16> Q;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Q[i * 4 + j] = impl_->Q(i, j);
        }
    }
    return Q;
}

bool KalmanDecoder::is_using_latent_space() const {
    return impl_->use_latent;
}

size_t KalmanDecoder::get_latent_dim() const {
    return impl_->latent_dim;
}

const PCAProjector* KalmanDecoder::get_pca_projector() const {
    return impl_->pca.get();
}

void KalmanDecoder::set_pca_from_checkpoint(
    std::span<const float> mean,
    std::span<const float> components,
    size_t n_features,
    size_t n_components
) {
    // Create a new PCA projector and manually set its state
    PCAProjector::Config pca_config;
    pca_config.n_components = n_components;
    
    impl_->pca = std::make_unique<PCAProjector>(pca_config);
    
    // Build training data from the components to fit the PCA
    // This is a workaround since we don't have direct access to PCA internals
    // In a production system, we'd add set_components() to PCAProjector
    
    // For now, we'll set the latent mode but mark that PCA needs proper restoration
    impl_->use_latent = true;
    impl_->latent_dim = n_components;
    
    // TODO: Add proper PCA state restoration via PCAProjector::load_state()
}

KalmanDecoder::CalibrationMetadata KalmanDecoder::get_calibration_metadata() const {
    CalibrationMetadata meta;
    meta.calibration_samples = impl_->calibration_samples;
    meta.r2_score = impl_->last_r2_score;
    meta.cv_score = impl_->last_cv_score;
    meta.ridge_lambda = impl_->last_lambda;
    return meta;
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

const ChannelConfig& KalmanDecoder::channel_config() const {
    return config_.channel_config;
}

size_t KalmanDecoder::num_channels() const {
    return impl_->num_channels;
}

// ============================================================================
// LinearDecoder Implementation
// ============================================================================

LinearDecoder::LinearDecoder(const Config& config)
    : config_(config)
    , running_mean_(config.channel_config.num_channels)
    , running_std_(config.channel_config.num_channels)
{
    // Initialize weights if empty
    if (config_.weights_x.empty()) {
        config_.weights_x.resize(config_.channel_config.num_channels, 0.0f);
    }
    if (config_.weights_y.empty()) {
        config_.weights_y.resize(config_.channel_config.num_channels, 0.0f);
    }
    
    for (size_t i = 0; i < running_std_.size(); ++i) {
        running_std_[i] = 1.0f;
    }
}

LinearDecoder::LinearDecoder(const ChannelConfig& channel_config)
    : LinearDecoder(Config(channel_config)) {}

LinearDecoder::LinearDecoder() {
    Config cfg;
    *this = LinearDecoder(cfg);
}

// Primary decode with SpikeData
DecoderOutput LinearDecoder::decode(const SpikeData& spike_data) {
    return decode(spike_data.span());
}

// Span-based decode
DecoderOutput LinearDecoder::decode(std::span<const float> spike_counts) {
    auto start = Clock::now();
    
    const size_t n = std::min(spike_counts.size(), num_channels());
    SpikeData normalized(n);
    
    // Normalize if enabled
    if (config_.normalize_input && sample_count_ > 10) {
        for (size_t i = 0; i < n; ++i) {
            float std = running_std_[i];
            if (std < 1e-6f) std = 1.0f;
            normalized[i] = (spike_counts[i] - running_mean_[i]) / std;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            normalized[i] = spike_counts[i];
        }
    }
    
    // Update running statistics (Welford's algorithm)
    for (size_t i = 0; i < n; ++i) {
        float delta = spike_counts[i] - running_mean_[i];
        running_mean_[i] += delta / static_cast<float>(sample_count_ + 1);
        float delta2 = spike_counts[i] - running_mean_[i];
        float var = running_std_[i] * running_std_[i];
        var += (delta * delta2 - var) / static_cast<float>(sample_count_ + 1);
        running_std_[i] = std::sqrt(var);
    }
    sample_count_++;
    
    // Apply linear decoder
    float pos_x = config_.bias_x;
    float pos_y = config_.bias_y;
    for (size_t i = 0; i < n; ++i) {
        pos_x += normalized[i] * config_.weights_x[i];
        pos_y += normalized[i] * config_.weights_y[i];
    }
    
    DecoderOutput output;
    output.position = {pos_x, pos_y};
    output.velocity = {0.0f, 0.0f};
    output.confidence = 1.0f;
    output.processing_time = Clock::now() - start;
    
    return output;
}

// Legacy decode methods
DecoderOutput LinearDecoder::decode(const SpikeCountArray& spike_counts) {
    const size_t n = std::min(spike_counts.size(), num_channels());
    SpikeData data(n);
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(spike_counts[i]);
    }
    return decode(data);
}

DecoderOutput LinearDecoder::decode(const AlignedSpikeData& spike_counts) {
    const size_t n = std::min(spike_counts.size(), num_channels());
    return decode(std::span<const float>(spike_counts.data(), n));
}

void LinearDecoder::train(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& positions
) {
    const size_t n_channels = num_channels();
    const size_t cols_to_use = std::min(static_cast<size_t>(neural_data.cols()), n_channels);
    
    // Compute normalization
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        running_mean_[ch] = neural_data.col(ch).mean();
        float variance = (neural_data.col(ch).array() - running_mean_[ch]).square().mean();
        running_std_[ch] = std::sqrt(variance);
        if (running_std_[ch] < 1e-6f) running_std_[ch] = 1.0f;
    }
    
    // Normalize
    Eigen::MatrixXf X(neural_data.rows(), cols_to_use + 1);
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        X.col(ch) = (neural_data.col(ch).array() - running_mean_[ch]) / running_std_[ch];
    }
    X.col(cols_to_use).setOnes();  // Bias term
    
    // Least squares: W = (X^T X)^-1 X^T Y
    Eigen::MatrixXf XtX = X.transpose() * X;
    Eigen::MatrixXf XtY = X.transpose() * positions;
    Eigen::MatrixXf W = XtX.ldlt().solve(XtY);
    
    // Extract weights
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        config_.weights_x[ch] = W(ch, 0);
        config_.weights_y[ch] = W(ch, 1);
    }
    config_.bias_x = W(cols_to_use, 0);
    config_.bias_y = W(cols_to_use, 1);
    
    sample_count_ = neural_data.rows();
}

void LinearDecoder::reset() {
    std::fill(config_.weights_x.begin(), config_.weights_x.end(), 0.0f);
    std::fill(config_.weights_y.begin(), config_.weights_y.end(), 0.0f);
    config_.bias_x = 0.0f;
    config_.bias_y = 0.0f;
    running_mean_.zero();
    for (size_t i = 0; i < running_std_.size(); ++i) {
        running_std_[i] = 1.0f;
    }
    sample_count_ = 0;
}

// ============================================================================
// VelocityKalmanDecoder Implementation (Dynamic channels)
// ============================================================================

struct VelocityKalmanDecoder::Impl {
    size_t num_channels = 142;  // Default, configurable
    
    Eigen::Vector2f state;
    Eigen::Matrix2f covariance;
    Eigen::MatrixXf H;  // Observation model [num_channels x 2]
    
    SpikeData spike_mean;
    SpikeData spike_std;
    
    explicit Impl(size_t channels = 142) 
        : num_channels(channels)
        , spike_mean(channels)
        , spike_std(channels)
        , H(Eigen::MatrixXf::Zero(channels, 2))
    {
        state.setZero();
        covariance.setIdentity();
        for (size_t i = 0; i < channels; ++i) {
            spike_std[i] = 1.0f;
        }
    }
};

VelocityKalmanDecoder::VelocityKalmanDecoder(const Config& config)
    : impl_(std::make_unique<Impl>(142))  // Default to MC_Maze for backward compat
    , config_(config) {}

DecoderOutput VelocityKalmanDecoder::decode(const SpikeCountArray& spike_counts) {
    auto start = Clock::now();
    
    const size_t n = std::min(spike_counts.size(), impl_->num_channels);
    
    // Normalize spikes
    SpikeData normalized(n);
    for (size_t i = 0; i < n; ++i) {
        float std = impl_->spike_std[i];
        if (std < 1e-6f) std = 1.0f;
        normalized[i] = (static_cast<float>(spike_counts[i]) - impl_->spike_mean[i]) / std;
    }
    
    // Predict - constant velocity model
    Eigen::Vector2f v_pred = impl_->state;
    Eigen::Matrix2f P_pred = impl_->covariance + config_.process_noise;
    
    // Update
    Eigen::Map<const Eigen::VectorXf> z(normalized.data(), n);
    
    Eigen::VectorXf z_pred = impl_->H.topRows(n) * v_pred;
    Eigen::VectorXf y = z - z_pred;
    
    Eigen::MatrixXf H_n = impl_->H.topRows(n);
    Eigen::MatrixXf S = H_n * P_pred * H_n.transpose() + 
        Eigen::MatrixXf::Identity(n, n) * config_.measurement_noise;
    
    Eigen::MatrixXf K = P_pred * H_n.transpose() * S.inverse();
    
    impl_->state = v_pred + K * y;
    impl_->covariance = (Eigen::Matrix2f::Identity() - K * H_n) * P_pred;
    
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
    const size_t n_channels = impl_->num_channels;
    const size_t cols_to_use = std::min(static_cast<size_t>(neural_data.cols()), n_channels);
    
    // Resize if needed
    if (cols_to_use != n_channels) {
        impl_ = std::make_unique<Impl>(cols_to_use);
    }
    
    // Compute normalization
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        impl_->spike_mean[ch] = neural_data.col(ch).mean();
        float variance = (neural_data.col(ch).array() - impl_->spike_mean[ch]).square().mean();
        impl_->spike_std[ch] = std::sqrt(variance);
        if (impl_->spike_std[ch] < 1e-6f) impl_->spike_std[ch] = 1.0f;
    }
    
    // Train observation model
    Eigen::MatrixXf X = velocities;  // [N x 2]
    Eigen::MatrixXf Z = neural_data.leftCols(cols_to_use);
    for (size_t ch = 0; ch < cols_to_use; ++ch) {
        Z.col(ch) = (Z.col(ch).array() - impl_->spike_mean[ch]) / impl_->spike_std[ch];
    }
    
    auto XtX = X.transpose() * X;
    auto XtZ = X.transpose() * Z;
    Eigen::MatrixXf H_temp = XtX.ldlt().solve(XtZ);  // [2 x cols_to_use]
    impl_->H = H_temp.transpose();  // [cols_to_use x 2]
}

}  // namespace phantomcore
