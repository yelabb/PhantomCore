#include "phantomcore/adaptive_decoder.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <deque>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace phantomcore {

// ============================================================================
// FiringRateMonitor Implementation
// ============================================================================

struct FiringRateMonitor::Impl {
    FiringRateMonitor::Config config;
    
    // Running statistics
    std::deque<std::vector<float>> recent_samples;
    std::vector<float> running_mean;
    std::vector<float> running_var;
    
    // Baseline statistics
    std::vector<float> baseline_mean;
    std::vector<float> baseline_var;
    bool baseline_established = false;
    size_t baseline_sample_count = 0;
    
    Impl(const Config& cfg) : config(cfg) {
        running_mean.resize(cfg.num_channels, 0.0f);
        running_var.resize(cfg.num_channels, 1.0f);
        baseline_mean.resize(cfg.num_channels, 0.0f);
        baseline_var.resize(cfg.num_channels, 1.0f);
    }
    
    void update_running_stats() {
        if (recent_samples.empty()) return;
        
        size_t n = recent_samples.size();
        for (size_t ch = 0; ch < config.num_channels; ++ch) {
            float sum = 0.0f;
            float sum_sq = 0.0f;
            for (const auto& sample : recent_samples) {
                sum += sample[ch];
                sum_sq += sample[ch] * sample[ch];
            }
            running_mean[ch] = sum / static_cast<float>(n);
            running_var[ch] = std::max(0.001f, sum_sq / static_cast<float>(n) - running_mean[ch] * running_mean[ch]);
        }
    }
};

FiringRateMonitor::FiringRateMonitor(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

FiringRateMonitor::~FiringRateMonitor() = default;

void FiringRateMonitor::observe(std::span<const float> spike_counts) {
    std::vector<float> sample(spike_counts.begin(), spike_counts.end());
    impl_->recent_samples.push_back(std::move(sample));
    
    // Maintain window size
    while (impl_->recent_samples.size() > impl_->config.window_size) {
        impl_->recent_samples.pop_front();
    }
    
    impl_->update_running_stats();
    
    // Build baseline if not established
    if (!impl_->baseline_established) {
        impl_->baseline_sample_count++;
        if (impl_->baseline_sample_count >= impl_->config.baseline_window) {
            reset_baseline();
        }
    }
}

float FiringRateMonitor::compute_kl_divergence() const {
    if (!impl_->baseline_established) return 0.0f;
    
    // Compute KL divergence assuming Gaussian distributions
    // KL(P||Q) = 0.5 * (log(var_q/var_p) + (var_p + (mu_p - mu_q)^2) / var_q - 1)
    float total_kl = 0.0f;
    
    for (size_t ch = 0; ch < impl_->config.num_channels; ++ch) {
        float mu_p = impl_->running_mean[ch];
        float var_p = impl_->running_var[ch];
        float mu_q = impl_->baseline_mean[ch];
        float var_q = impl_->baseline_var[ch];
        
        float kl = 0.5f * (std::log(var_q / var_p) + 
                          (var_p + (mu_p - mu_q) * (mu_p - mu_q)) / var_q - 1.0f);
        total_kl += std::max(0.0f, kl);  // KL is always non-negative
    }
    
    return total_kl / static_cast<float>(impl_->config.num_channels);
}

std::vector<float> FiringRateMonitor::get_rate_changes() const {
    std::vector<float> changes(impl_->config.num_channels);
    
    if (!impl_->baseline_established) {
        std::fill(changes.begin(), changes.end(), 0.0f);
        return changes;
    }
    
    for (size_t ch = 0; ch < impl_->config.num_channels; ++ch) {
        float baseline_std = std::sqrt(impl_->baseline_var[ch]);
        changes[ch] = (impl_->running_mean[ch] - impl_->baseline_mean[ch]) / 
                      std::max(0.001f, baseline_std);
    }
    
    return changes;
}

std::vector<size_t> FiringRateMonitor::get_drifted_channels(float threshold) const {
    std::vector<size_t> drifted;
    auto changes = get_rate_changes();
    
    for (size_t ch = 0; ch < changes.size(); ++ch) {
        if (std::abs(changes[ch]) > threshold) {
            drifted.push_back(ch);
        }
    }
    
    return drifted;
}

void FiringRateMonitor::reset_baseline() {
    impl_->baseline_mean = impl_->running_mean;
    impl_->baseline_var = impl_->running_var;
    impl_->baseline_established = true;
}

bool FiringRateMonitor::has_baseline() const {
    return impl_->baseline_established;
}

// ============================================================================
// SupervisedBuffer Implementation
// ============================================================================

struct SupervisedBuffer::Impl {
    std::deque<Sample> buffer;
    size_t capacity;
    
    Impl(size_t cap) : capacity(cap) {}
};

SupervisedBuffer::SupervisedBuffer(size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}

SupervisedBuffer::~SupervisedBuffer() = default;

void SupervisedBuffer::push_neural(const SpikeData& data, Timestamp ts) {
    Sample sample;
    sample.neural_data = SpikeData(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        sample.neural_data[i] = data[i];
    }
    sample.timestamp = ts;
    sample.ground_truth = std::nullopt;
    sample.used_for_update = false;
    
    impl_->buffer.push_back(std::move(sample));
    
    while (impl_->buffer.size() > impl_->capacity) {
        impl_->buffer.pop_front();
    }
}

bool SupervisedBuffer::match_ground_truth(const Vec4& ground_truth, Timestamp ts, Duration tolerance) {
    int64_t tolerance_ns = tolerance.count();
    
    for (auto& sample : impl_->buffer) {
        if (sample.ground_truth.has_value()) continue;
        
        auto diff = std::chrono::duration_cast<std::chrono::nanoseconds>(ts - sample.timestamp);
        if (std::abs(diff.count()) <= tolerance_ns) {
            sample.ground_truth = ground_truth;
            return true;
        }
    }
    
    return false;
}

std::vector<std::pair<SpikeData, Vec4>> SupervisedBuffer::get_matched_samples(size_t max_count) {
    std::vector<std::pair<SpikeData, Vec4>> result;
    
    for (auto& sample : impl_->buffer) {
        if (sample.ground_truth.has_value() && !sample.used_for_update) {
            SpikeData copy(sample.neural_data.size());
            for (size_t i = 0; i < sample.neural_data.size(); ++i) {
                copy[i] = sample.neural_data[i];
            }
            result.emplace_back(std::move(copy), *sample.ground_truth);
            sample.used_for_update = true;
            
            if (max_count > 0 && result.size() >= max_count) break;
        }
    }
    
    return result;
}

size_t SupervisedBuffer::unmatched_count() const {
    size_t count = 0;
    for (const auto& sample : impl_->buffer) {
        if (!sample.ground_truth.has_value()) count++;
    }
    return count;
}

void SupervisedBuffer::clear() {
    impl_->buffer.clear();
}

// ============================================================================
// AdaptiveDecoder Implementation
// ============================================================================

struct AdaptiveDecoder::Impl {
    AdaptiveDecoder::Config config;
    
    // Base decoder
    std::unique_ptr<KalmanDecoder> decoder;
    
    // RLS state
    Eigen::MatrixXf P_rls;           // Inverse correlation matrix
    Eigen::MatrixXf H_current;       // Current observation weights
    Eigen::MatrixXf H_baseline;      // Baseline weights for regularization
    bool rls_initialized = false;
    
    // Rollback history
    std::deque<Eigen::MatrixXf> weight_history;
    
    // Adaptation state
    bool adaptation_enabled = true;
    bool frozen = false;
    float current_learning_rate;
    float current_forgetting_factor;
    
    // Drift detection
    std::unique_ptr<FiringRateMonitor> firing_rate_monitor;
    DriftDetectionResult last_drift_result;
    DriftCallback drift_callback;
    
    // Supervised data matching
    SupervisedBuffer supervised_buffer;
    
    // Prediction tracking for error computation
    std::deque<Vec4> recent_predictions;
    std::deque<Vec4> recent_ground_truths;
    float baseline_error = 0.0f;
    
    // Statistics
    AdaptationStats stats;
    LatencyTracker update_latency;
    Timestamp calibration_time;
    size_t samples_since_calibration = 0;
    
    Impl(const Config& cfg) 
        : config(cfg)
        , supervised_buffer(1000)
    {
        current_learning_rate = cfg.adaptive.learning_rate;
        current_forgetting_factor = cfg.adaptive.forgetting_factor;
        
        decoder = std::make_unique<KalmanDecoder>(cfg.kalman_config);
        
        FiringRateMonitor::Config fr_config;
        fr_config.num_channels = cfg.kalman_config.channel_config.num_channels;
        fr_config.window_size = cfg.adaptive.drift_window_samples;
        firing_rate_monitor = std::make_unique<FiringRateMonitor>(fr_config);
    }
    
    void initialize_rls(size_t n_features) {
        // Initialize inverse correlation matrix as scaled identity
        float delta = 1.0f / config.adaptive.learning_rate;
        P_rls = Eigen::MatrixXf::Identity(n_features, n_features) * delta;
        rls_initialized = true;
    }
    
    void perform_rls_update(
        const Eigen::VectorXf& features,
        const Eigen::Vector4f& target,
        const Eigen::Vector4f& prediction
    ) {
        if (!rls_initialized) return;
        
        auto start = Clock::now();
        
        float lambda = current_forgetting_factor;
        
        // Compute error
        Eigen::Vector4f error = target - prediction;
        
        // Compute gain: k = P * x / (λ + x' * P * x)
        Eigen::VectorXf Px = P_rls * features;
        float denom = lambda + features.dot(Px);
        Eigen::VectorXf k = Px / denom;
        
        // Update weights: H = H + k * error'
        for (int i = 0; i < 4; ++i) {
            H_current.col(i) += k * error(i);
        }
        
        // Apply constraint: limit weight change
        if (config.adaptive.max_weight_change_per_update > 0) {
            Eigen::MatrixXf delta = H_current - H_baseline;
            float norm = delta.norm();
            float max_norm = config.adaptive.max_divergence_from_baseline;
            if (norm > max_norm) {
                H_current = H_baseline + delta * (max_norm / norm);
            }
        }
        
        // Baseline regularization: pull towards baseline
        if (config.adaptive.preserve_baseline_weights) {
            float reg = config.adaptive.baseline_regularization;
            H_current = (1.0f - reg) * H_current + reg * H_baseline;
        }
        
        // Update inverse correlation: P = (P - k * x' * P) / λ
        Eigen::MatrixXf kxP = k * (features.transpose() * P_rls);
        P_rls = (P_rls - kxP) / lambda;
        
        // Save to history for rollback
        if (config.adaptive.enable_rollback) {
            weight_history.push_back(H_current);
            while (weight_history.size() > config.adaptive.rollback_window) {
                weight_history.pop_front();
            }
        }
        
        update_latency.record(Clock::now() - start);
        stats.supervised_updates++;
    }
    
    DriftDetectionResult compute_drift() {
        DriftDetectionResult result;
        result.samples_since_calibration = samples_since_calibration;
        result.time_since_calibration = Clock::now() - calibration_time;
        
        if (config.adaptive.drift_method == DriftDetectionMethod::None) {
            return result;
        }
        
        // Compute firing rate divergence
        if (firing_rate_monitor->has_baseline()) {
            result.firing_rate_divergence = firing_rate_monitor->compute_kl_divergence();
        }
        
        // Compute prediction error ratio
        if (!recent_predictions.empty() && !recent_ground_truths.empty()) {
            float recent_error = 0.0f;
            size_t n = std::min(recent_predictions.size(), recent_ground_truths.size());
            for (size_t i = 0; i < n; ++i) {
                Vec4 diff = {
                    recent_predictions[i].x - recent_ground_truths[i].x,
                    recent_predictions[i].y - recent_ground_truths[i].y,
                    recent_predictions[i].vx - recent_ground_truths[i].vx,
                    recent_predictions[i].vy - recent_ground_truths[i].vy
                };
                recent_error += std::sqrt(diff.x*diff.x + diff.y*diff.y + 
                                         diff.vx*diff.vx + diff.vy*diff.vy);
            }
            recent_error /= static_cast<float>(n);
            
            if (baseline_error > 0) {
                result.prediction_error_ratio = recent_error / baseline_error;
            }
            stats.recent_prediction_error = recent_error;
        }
        
        // Compute covariance distance (simplified: weight drift)
        if (rls_initialized) {
            result.covariance_distance = (H_current - H_baseline).norm() / 
                                         std::max(1.0f, H_baseline.norm());
        }
        
        // Combine metrics for overall drift score
        switch (config.adaptive.drift_method) {
            case DriftDetectionMethod::FiringRateKL:
                result.drift_score = result.firing_rate_divergence;
                break;
            case DriftDetectionMethod::DecodingError:
                result.drift_score = std::max(0.0f, result.prediction_error_ratio - 1.0f);
                break;
            case DriftDetectionMethod::CovarianceShift:
                result.drift_score = result.covariance_distance;
                break;
            case DriftDetectionMethod::HybridMultiMetric:
            default:
                result.drift_score = 0.4f * result.firing_rate_divergence +
                                    0.4f * std::max(0.0f, result.prediction_error_ratio - 1.0f) +
                                    0.2f * result.covariance_distance;
                break;
        }
        
        // Determine drift status and recommended action
        result.drift_detected = result.drift_score > config.adaptive.drift_threshold;
        
        if (result.drift_score > config.adaptive.recalibration_threshold) {
            result.recommended_action = DriftDetectionResult::Action::Recalibrate;
        } else if (result.drift_score > config.adaptive.drift_threshold) {
            result.recommended_action = DriftDetectionResult::Action::IncreaseLearning;
        }
        
        return result;
    }
};

AdaptiveDecoder::AdaptiveDecoder(const Config& config)
    : impl_(std::make_unique<Impl>(config)) {}

AdaptiveDecoder::~AdaptiveDecoder() = default;

AdaptiveDecoder::AdaptiveDecoder(AdaptiveDecoder&&) noexcept = default;
AdaptiveDecoder& AdaptiveDecoder::operator=(AdaptiveDecoder&&) noexcept = default;

DecoderOutput AdaptiveDecoder::decode(const SpikeData& spike_data) {
    impl_->samples_since_calibration++;
    impl_->stats.total_decodes++;
    
    // Update firing rate monitor
    impl_->firing_rate_monitor->observe(
        std::span<const float>(spike_data.data(), spike_data.size())
    );
    
    // Store for potential supervised update
    impl_->supervised_buffer.push_neural(spike_data, Clock::now());
    
    return impl_->decoder->decode(spike_data);
}

DecoderOutput AdaptiveDecoder::decode(std::span<const float> spike_counts) {
    SpikeData data(spike_counts.size());
    for (size_t i = 0; i < spike_counts.size(); ++i) {
        data[i] = spike_counts[i];
    }
    return decode(data);
}

DecoderOutput AdaptiveDecoder::decode_and_update(
    const SpikeData& spike_data,
    const Vec4& ground_truth
) {
    // First decode
    auto output = decode(spike_data);
    
    // Skip update if frozen or disabled
    if (impl_->frozen || !impl_->adaptation_enabled) {
        return output;
    }
    
    // Check update strategy
    bool should_update = false;
    
    switch (impl_->config.adaptive.strategy) {
        case AdaptiveConfig::UpdateStrategy::EveryTimestep:
            should_update = true;
            break;
            
        case AdaptiveConfig::UpdateStrategy::ErrorThreshold: {
            float error = std::sqrt(
                (output.position.x - ground_truth.x) * (output.position.x - ground_truth.x) +
                (output.position.y - ground_truth.y) * (output.position.y - ground_truth.y)
            );
            should_update = error > impl_->config.adaptive.error_threshold;
            break;
        }
        
        case AdaptiveConfig::UpdateStrategy::BatchedUpdates:
        case AdaptiveConfig::UpdateStrategy::TimeBased:
            // Handled elsewhere
            should_update = false;
            break;
    }
    
    if (should_update && impl_->rls_initialized) {
        Eigen::VectorXf features(spike_data.size());
        for (size_t i = 0; i < spike_data.size(); ++i) {
            features(static_cast<Eigen::Index>(i)) = spike_data[i];
        }
        
        Eigen::Vector4f target;
        target << ground_truth.x, ground_truth.y, ground_truth.vx, ground_truth.vy;
        
        Eigen::Vector4f prediction;
        prediction << output.position.x, output.position.y, 
                     output.velocity.vx, output.velocity.vy;
        
        impl_->perform_rls_update(features, target, prediction);
        
        // Track for drift detection
        impl_->recent_predictions.push_back({output.position.x, output.position.y,
                                             output.velocity.vx, output.velocity.vy});
        impl_->recent_ground_truths.push_back(ground_truth);
        
        while (impl_->recent_predictions.size() > impl_->config.adaptive.drift_window_samples) {
            impl_->recent_predictions.pop_front();
            impl_->recent_ground_truths.pop_front();
        }
    } else {
        impl_->stats.skipped_updates++;
    }
    
    // Check for drift
    impl_->last_drift_result = impl_->compute_drift();
    if (impl_->last_drift_result.drift_detected && impl_->drift_callback) {
        impl_->drift_callback(impl_->last_drift_result);
    }
    
    // Adjust learning rate based on drift
    if (impl_->config.adaptive.adaptive_learning_rate) {
        if (impl_->last_drift_result.drift_detected) {
            impl_->current_learning_rate = std::min(
                impl_->config.adaptive.max_learning_rate,
                impl_->current_learning_rate * 1.1f
            );
        } else {
            impl_->current_learning_rate = std::max(
                impl_->config.adaptive.min_learning_rate,
                impl_->current_learning_rate * 0.99f
            );
        }
        impl_->stats.learning_rate_current = impl_->current_learning_rate;
    }
    
    return output;
}

void AdaptiveDecoder::provide_ground_truth(const Vec4& ground_truth, Timestamp timestamp) {
    Duration tolerance = std::chrono::milliseconds(50);  // 50ms tolerance
    impl_->supervised_buffer.match_ground_truth(ground_truth, timestamp, tolerance);
    
    // Process any matched samples
    auto matched = impl_->supervised_buffer.get_matched_samples(impl_->config.adaptive.batch_size);
    
    for (const auto& [neural, target] : matched) {
        if (impl_->rls_initialized) {
            Eigen::VectorXf features(neural.size());
            for (size_t i = 0; i < neural.size(); ++i) {
                features(static_cast<Eigen::Index>(i)) = neural[i];
            }
            
            // Compute prediction for this sample
            SpikeData temp(neural.size());
            for (size_t i = 0; i < neural.size(); ++i) {
                temp[i] = neural[i];
            }
            auto output = impl_->decoder->decode(temp);
            
            Eigen::Vector4f target_vec;
            target_vec << target.x, target.y, target.vx, target.vy;
            
            Eigen::Vector4f pred_vec;
            pred_vec << output.position.x, output.position.y,
                       output.velocity.vx, output.velocity.vy;
            
            impl_->perform_rls_update(features, target_vec, pred_vec);
        }
    }
}

DecoderOutput AdaptiveDecoder::predict() {
    return impl_->decoder->predict();
}

void AdaptiveDecoder::set_adaptation_enabled(bool enabled) {
    impl_->adaptation_enabled = enabled;
}

bool AdaptiveDecoder::is_adaptation_enabled() const {
    return impl_->adaptation_enabled;
}

void AdaptiveDecoder::set_learning_rate(float rate) {
    impl_->current_learning_rate = std::clamp(
        rate,
        impl_->config.adaptive.min_learning_rate,
        impl_->config.adaptive.max_learning_rate
    );
}

float AdaptiveDecoder::get_learning_rate() const {
    return impl_->current_learning_rate;
}

void AdaptiveDecoder::set_forgetting_factor(float lambda) {
    impl_->current_forgetting_factor = std::clamp(
        lambda,
        impl_->config.adaptive.min_forgetting_factor,
        impl_->config.adaptive.max_forgetting_factor
    );
}

float AdaptiveDecoder::get_forgetting_factor() const {
    return impl_->current_forgetting_factor;
}

void AdaptiveDecoder::freeze() {
    impl_->frozen = true;
}

void AdaptiveDecoder::unfreeze() {
    impl_->frozen = false;
}

void AdaptiveDecoder::force_update(
    const Eigen::MatrixXf& neural_batch,
    const Eigen::MatrixXf& kinematics_batch
) {
    if (!impl_->rls_initialized) return;
    
    Eigen::Index n_samples = neural_batch.rows();
    for (Eigen::Index i = 0; i < n_samples; ++i) {
        Eigen::VectorXf features = neural_batch.row(i).transpose();
        Eigen::Vector4f target = kinematics_batch.row(i).transpose();
        
        // Simple prediction using current weights
        Eigen::Vector4f prediction = impl_->H_current.transpose() * features;
        
        impl_->perform_rls_update(features, target, prediction);
    }
}

DriftDetectionResult AdaptiveDecoder::get_drift_status() const {
    return impl_->last_drift_result;
}

DriftDetectionResult AdaptiveDecoder::analyze_drift() {
    impl_->last_drift_result = impl_->compute_drift();
    return impl_->last_drift_result;
}

void AdaptiveDecoder::on_drift_detected(DriftCallback callback) {
    impl_->drift_callback = std::move(callback);
}

void AdaptiveDecoder::reset_drift_baseline() {
    impl_->firing_rate_monitor->reset_baseline();
    impl_->baseline_error = impl_->stats.recent_prediction_error;
    impl_->recent_predictions.clear();
    impl_->recent_ground_truths.clear();
}

bool AdaptiveDecoder::rollback(size_t steps) {
    if (impl_->weight_history.empty()) return false;
    
    size_t actual_steps = std::min(steps, impl_->weight_history.size());
    
    for (size_t i = 0; i < actual_steps; ++i) {
        if (!impl_->weight_history.empty()) {
            impl_->weight_history.pop_back();
        }
    }
    
    if (!impl_->weight_history.empty()) {
        impl_->H_current = impl_->weight_history.back();
    } else {
        impl_->H_current = impl_->H_baseline;
    }
    
    impl_->stats.rollbacks_performed++;
    return true;
}

void AdaptiveDecoder::rollback_to_baseline() {
    impl_->H_current = impl_->H_baseline;
    impl_->weight_history.clear();
    
    // Reinitialize RLS
    size_t n_features = impl_->H_current.rows();
    float delta = 1.0f / impl_->config.adaptive.learning_rate;
    impl_->P_rls = Eigen::MatrixXf::Identity(n_features, n_features) * delta;
    
    impl_->stats.rollbacks_performed++;
}

size_t AdaptiveDecoder::available_rollback_steps() const {
    return impl_->weight_history.size();
}

void AdaptiveDecoder::commit_as_baseline() {
    impl_->H_baseline = impl_->H_current;
    impl_->weight_history.clear();
    reset_drift_baseline();
}

KalmanDecoder::CalibrationResult AdaptiveDecoder::calibrate(
    const Eigen::MatrixXf& neural_data,
    const Eigen::MatrixXf& kinematics
) {
    auto result = impl_->decoder->calibrate(neural_data, kinematics);
    
    if (result.success) {
        // Initialize RLS state
        size_t n_features = static_cast<size_t>(neural_data.cols());
        impl_->initialize_rls(n_features);
        
        // Store baseline weights
        auto weights = impl_->decoder->save_weights();
        impl_->H_baseline = Eigen::Map<Eigen::MatrixXf>(
            weights.data(), static_cast<Eigen::Index>(n_features), 4
        );
        impl_->H_current = impl_->H_baseline;
        
        // Reset tracking
        impl_->calibration_time = Clock::now();
        impl_->samples_since_calibration = 0;
        impl_->weight_history.clear();
        
        // Initialize baseline error
        float total_error = 0.0f;
        for (Eigen::Index i = 0; i < neural_data.rows(); ++i) {
            Eigen::Vector4f pred = impl_->H_baseline.transpose() * neural_data.row(i).transpose();
            Eigen::Vector4f target = kinematics.row(i).transpose();
            total_error += (pred - target).norm();
        }
        impl_->baseline_error = total_error / static_cast<float>(neural_data.rows());
        impl_->stats.baseline_prediction_error = impl_->baseline_error;
        
        // Reset drift baseline
        impl_->firing_rate_monitor->reset_baseline();
    }
    
    return result;
}

void AdaptiveDecoder::reset() {
    impl_->decoder->reset();
    impl_->weight_history.clear();
    impl_->recent_predictions.clear();
    impl_->recent_ground_truths.clear();
    impl_->supervised_buffer.clear();
    impl_->samples_since_calibration = 0;
    
    if (impl_->rls_initialized) {
        impl_->H_current = impl_->H_baseline;
        size_t n_features = impl_->H_current.rows();
        float delta = 1.0f / impl_->config.adaptive.learning_rate;
        impl_->P_rls = Eigen::MatrixXf::Identity(n_features, n_features) * delta;
    }
}

AdaptiveDecoder::AdaptationStats AdaptiveDecoder::get_stats() const {
    auto stats = impl_->stats;
    stats.update_latency = impl_->update_latency.get_stats();
    stats.learning_rate_current = impl_->current_learning_rate;
    stats.forgetting_factor_current = impl_->current_forgetting_factor;
    
    if (impl_->rls_initialized) {
        stats.weight_drift_from_baseline = (impl_->H_current - impl_->H_baseline).norm();
    }
    
    return stats;
}

void AdaptiveDecoder::reset_stats() {
    impl_->stats = AdaptationStats{};
    impl_->update_latency = LatencyTracker{};
}

const KalmanDecoder& AdaptiveDecoder::base_decoder() const {
    return *impl_->decoder;
}

KalmanDecoder::StateVector AdaptiveDecoder::get_state() const {
    return impl_->decoder->get_state();
}

}  // namespace phantomcore
