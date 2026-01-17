#include "phantomcore/model_checkpoint.hpp"
#include <chrono>
#include <iomanip>
#include <sstream>

namespace phantomcore {

// ============================================================================
// Create Checkpoint from Trained Decoder
// ============================================================================

ModelCheckpoint create_checkpoint(const KalmanDecoder& decoder) {
    ModelCheckpoint checkpoint;
    
    // === Channel Configuration ===
    checkpoint.channel_config = decoder.channel_config();
    
    // === Spike Normalization ===
    auto [mean, std] = decoder.get_normalization_params();
    checkpoint.spike_mean = std::move(mean);
    checkpoint.spike_std = std::move(std);
    
    // === Calibration Metadata ===
    auto meta = decoder.get_calibration_metadata();
    checkpoint.calibration_samples = meta.calibration_samples;
    checkpoint.calibration_r2_score = meta.r2_score;
    checkpoint.calibration_cv_score = meta.cv_score;
    checkpoint.ridge_lambda = meta.ridge_lambda;
    
    // === PCA State ===
    checkpoint.has_pca = decoder.is_using_latent_space();
    if (checkpoint.has_pca) {
        const PCAProjector* pca = decoder.get_pca_projector();
        if (pca && pca->is_fitted()) {
            auto pca_state = pca->get_serialized_state();
            checkpoint.pca_n_components = pca_state.n_components;
            checkpoint.pca_n_features = pca_state.n_features;
            checkpoint.pca_mean = std::move(pca_state.mean);
            checkpoint.pca_components = std::move(pca_state.components);
            checkpoint.pca_explained_variance = std::move(pca_state.explained_var);
            checkpoint.pca_total_variance_explained = pca_state.total_variance_explained;
        }
    }
    
    // === Kalman State ===
    auto state = decoder.get_state();
    for (int i = 0; i < 4; ++i) {
        checkpoint.kalman_state[i] = state(i);
    }
    
    auto cov = decoder.get_covariance();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            checkpoint.kalman_covariance[i * 4 + j] = cov(i, j);
        }
    }
    
    checkpoint.kalman_A = decoder.get_state_transition();
    checkpoint.kalman_Q = decoder.get_process_noise();
    checkpoint.kalman_H = decoder.get_observation_matrix();
    checkpoint.kalman_H_latent = decoder.get_latent_observation_matrix();
    
    // === Timestamp ===
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    checkpoint.created_timestamp = oss.str();
    
    return checkpoint;
}

// ============================================================================
// Restore Decoder from Checkpoint
// ============================================================================

void restore_from_checkpoint(KalmanDecoder& decoder, const ModelCheckpoint& checkpoint) {
    // Validate checkpoint
    if (!checkpoint.validate()) {
        throw std::runtime_error("Invalid checkpoint: data size mismatch");
    }
    
    // Check channel count matches
    if (decoder.num_channels() != checkpoint.channel_config.num_channels) {
        throw std::runtime_error(
            "Channel count mismatch: decoder has " + 
            std::to_string(decoder.num_channels()) + 
            ", checkpoint has " + 
            std::to_string(checkpoint.channel_config.num_channels)
        );
    }
    
    // === Spike Normalization ===
    decoder.set_normalization_params(checkpoint.spike_mean, checkpoint.spike_std);
    
    // === Observation Matrix ===
    decoder.set_observation_matrix(
        checkpoint.kalman_H,
        checkpoint.channel_config.num_channels,
        4  // STATE_DIM
    );
    
    // === Latent Space (if enabled) ===
    if (checkpoint.has_pca && !checkpoint.kalman_H_latent.empty()) {
        decoder.set_latent_observation_matrix(
            checkpoint.kalman_H_latent,
            checkpoint.pca_n_components,
            4  // STATE_DIM
        );
        
        // Restore PCA state
        decoder.set_pca_from_checkpoint(
            checkpoint.pca_mean,
            checkpoint.pca_components,
            checkpoint.pca_n_features,
            checkpoint.pca_n_components
        );
    }
    
    // Note: Kalman state (x) and covariance (P) could also be restored,
    // but typically it's better to start fresh for a new session
}

} // namespace phantomcore
