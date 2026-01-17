#pragma once

#include "types.hpp"
#include "kalman_decoder.hpp"
#include "dimensionality_reduction.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <cstring>

namespace phantomcore {

/**
 * @brief Complete pipeline state for session persistence
 * 
 * This structure captures ALL necessary state to restore a trained
 * decoder pipeline, including:
 * - Channel configuration
 * - Spike normalization parameters (mean/std for z-score)
 * - PCA projection matrix and centering
 * - Kalman filter state and covariance
 * - Observation model weights
 * 
 * Binary format for efficient serialization:
 * [Header][ChannelConfig][Normalization][PCA][KalmanState][Metadata]
 */
struct ModelCheckpoint {
    // ========================================================================
    // Magic & Version
    // ========================================================================
    static constexpr uint32_t MAGIC = 0x50484D43;  // "PHMC" = PhantomCore Model Checkpoint
    static constexpr uint32_t VERSION = 1;
    
    // ========================================================================
    // Channel Configuration
    // ========================================================================
    ChannelConfig channel_config;
    
    // ========================================================================
    // Spike Normalization (Z-Score Parameters)
    // ========================================================================
    std::vector<float> spike_mean;     // Per-channel mean firing rate
    std::vector<float> spike_std;      // Per-channel standard deviation
    size_t calibration_samples = 0;    // Number of samples used for calibration
    
    // ========================================================================
    // PCA State (Dimensionality Reduction)
    // ========================================================================
    bool has_pca = false;
    size_t pca_n_components = 0;
    size_t pca_n_features = 0;
    std::vector<float> pca_mean;       // Feature means [n_features]
    std::vector<float> pca_components; // Principal components [n_features x n_components]
    std::vector<float> pca_explained_variance;  // Variance per component
    float pca_total_variance_explained = 0.0f;
    
    // ========================================================================
    // Kalman Filter State
    // ========================================================================
    std::array<float, 4> kalman_state = {};         // [x, y, vx, vy]
    std::array<float, 16> kalman_covariance = {};   // 4x4 flattened (row-major)
    std::array<float, 16> kalman_A = {};            // State transition matrix
    std::array<float, 16> kalman_Q = {};            // Process noise covariance
    std::vector<float> kalman_H;                    // Observation matrix [n_channels x 4]
    std::vector<float> kalman_H_latent;             // Latent observation matrix [n_components x 4]
    float measurement_noise_scale = 0.1f;
    
    // ========================================================================
    // Calibration Metadata
    // ========================================================================
    float calibration_r2_score = 0.0f;
    float calibration_cv_score = 0.0f;
    float ridge_lambda = 0.0f;
    std::string model_name;
    std::string created_timestamp;
    std::string notes;
    
    // ========================================================================
    // Serialization
    // ========================================================================
    
    /**
     * @brief Serialize checkpoint to binary buffer
     * @return Binary representation of the checkpoint
     */
    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buffer;
        
        // Reserve approximate size
        size_t est_size = 1024 + 
            spike_mean.size() * sizeof(float) * 2 +
            pca_components.size() * sizeof(float) +
            kalman_H.size() * sizeof(float);
        buffer.reserve(est_size);
        
        // Helper to append data
        auto append = [&buffer](const void* data, size_t size) {
            const uint8_t* ptr = static_cast<const uint8_t*>(data);
            buffer.insert(buffer.end(), ptr, ptr + size);
        };
        
        auto append_string = [&](const std::string& s) {
            uint32_t len = static_cast<uint32_t>(s.size());
            append(&len, sizeof(len));
            append(s.data(), s.size());
        };
        
        auto append_vector = [&](const auto& vec) {
            uint32_t size = static_cast<uint32_t>(vec.size());
            append(&size, sizeof(size));
            if (!vec.empty()) {
                append(vec.data(), vec.size() * sizeof(typename std::decay_t<decltype(vec)>::value_type));
            }
        };
        
        // === Header ===
        append(&MAGIC, sizeof(MAGIC));
        append(&VERSION, sizeof(VERSION));
        
        // === Channel Config ===
        append(&channel_config.num_channels, sizeof(channel_config.num_channels));
        uint8_t preset = static_cast<uint8_t>(channel_config.preset);
        append(&preset, sizeof(preset));
        append_string(channel_config.name);
        
        // === Spike Normalization ===
        append_vector(spike_mean);
        append_vector(spike_std);
        append(&calibration_samples, sizeof(calibration_samples));
        
        // === PCA ===
        uint8_t has_pca_byte = has_pca ? 1 : 0;
        append(&has_pca_byte, sizeof(has_pca_byte));
        if (has_pca) {
            append(&pca_n_components, sizeof(pca_n_components));
            append(&pca_n_features, sizeof(pca_n_features));
            append_vector(pca_mean);
            append_vector(pca_components);
            append_vector(pca_explained_variance);
            append(&pca_total_variance_explained, sizeof(pca_total_variance_explained));
        }
        
        // === Kalman State ===
        append(kalman_state.data(), kalman_state.size() * sizeof(float));
        append(kalman_covariance.data(), kalman_covariance.size() * sizeof(float));
        append(kalman_A.data(), kalman_A.size() * sizeof(float));
        append(kalman_Q.data(), kalman_Q.size() * sizeof(float));
        append_vector(kalman_H);
        append_vector(kalman_H_latent);
        append(&measurement_noise_scale, sizeof(measurement_noise_scale));
        
        // === Metadata ===
        append(&calibration_r2_score, sizeof(calibration_r2_score));
        append(&calibration_cv_score, sizeof(calibration_cv_score));
        append(&ridge_lambda, sizeof(ridge_lambda));
        append_string(model_name);
        append_string(created_timestamp);
        append_string(notes);
        
        return buffer;
    }
    
    /**
     * @brief Deserialize checkpoint from binary buffer
     * @param data Binary data
     * @return Deserialized checkpoint
     * @throws std::runtime_error on invalid data
     */
    static ModelCheckpoint deserialize(std::span<const uint8_t> data) {
        ModelCheckpoint checkpoint;
        size_t offset = 0;
        
        auto read = [&](void* dest, size_t size) {
            if (offset + size > data.size()) {
                throw std::runtime_error("Checkpoint data truncated");
            }
            std::memcpy(dest, data.data() + offset, size);
            offset += size;
        };
        
        auto read_string = [&]() -> std::string {
            uint32_t len;
            read(&len, sizeof(len));
            if (offset + len > data.size()) {
                throw std::runtime_error("String data truncated");
            }
            std::string s(reinterpret_cast<const char*>(data.data() + offset), len);
            offset += len;
            return s;
        };
        
        auto read_float_vector = [&]() -> std::vector<float> {
            uint32_t size;
            read(&size, sizeof(size));
            std::vector<float> vec(size);
            if (size > 0) {
                read(vec.data(), size * sizeof(float));
            }
            return vec;
        };
        
        // === Header ===
        uint32_t magic, version;
        read(&magic, sizeof(magic));
        if (magic != MAGIC) {
            throw std::runtime_error("Invalid checkpoint magic number");
        }
        read(&version, sizeof(version));
        if (version > VERSION) {
            throw std::runtime_error("Checkpoint version too new: " + std::to_string(version));
        }
        
        // === Channel Config ===
        read(&checkpoint.channel_config.num_channels, sizeof(checkpoint.channel_config.num_channels));
        uint8_t preset;
        read(&preset, sizeof(preset));
        checkpoint.channel_config.preset = static_cast<HardwarePreset>(preset);
        checkpoint.channel_config.name = read_string();
        
        // === Spike Normalization ===
        checkpoint.spike_mean = read_float_vector();
        checkpoint.spike_std = read_float_vector();
        read(&checkpoint.calibration_samples, sizeof(checkpoint.calibration_samples));
        
        // === PCA ===
        uint8_t has_pca_byte;
        read(&has_pca_byte, sizeof(has_pca_byte));
        checkpoint.has_pca = has_pca_byte != 0;
        if (checkpoint.has_pca) {
            read(&checkpoint.pca_n_components, sizeof(checkpoint.pca_n_components));
            read(&checkpoint.pca_n_features, sizeof(checkpoint.pca_n_features));
            checkpoint.pca_mean = read_float_vector();
            checkpoint.pca_components = read_float_vector();
            checkpoint.pca_explained_variance = read_float_vector();
            read(&checkpoint.pca_total_variance_explained, sizeof(checkpoint.pca_total_variance_explained));
        }
        
        // === Kalman State ===
        read(checkpoint.kalman_state.data(), checkpoint.kalman_state.size() * sizeof(float));
        read(checkpoint.kalman_covariance.data(), checkpoint.kalman_covariance.size() * sizeof(float));
        read(checkpoint.kalman_A.data(), checkpoint.kalman_A.size() * sizeof(float));
        read(checkpoint.kalman_Q.data(), checkpoint.kalman_Q.size() * sizeof(float));
        checkpoint.kalman_H = read_float_vector();
        checkpoint.kalman_H_latent = read_float_vector();
        read(&checkpoint.measurement_noise_scale, sizeof(checkpoint.measurement_noise_scale));
        
        // === Metadata ===
        read(&checkpoint.calibration_r2_score, sizeof(checkpoint.calibration_r2_score));
        read(&checkpoint.calibration_cv_score, sizeof(checkpoint.calibration_cv_score));
        read(&checkpoint.ridge_lambda, sizeof(checkpoint.ridge_lambda));
        checkpoint.model_name = read_string();
        checkpoint.created_timestamp = read_string();
        checkpoint.notes = read_string();
        
        return checkpoint;
    }
    
    /**
     * @brief Save checkpoint to file
     * @param filepath Path to output file
     */
    void save(const std::string& filepath) const {
        auto data = serialize();
        std::ofstream file(filepath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Failed to open file for writing: " + filepath);
        }
        file.write(reinterpret_cast<const char*>(data.data()), data.size());
    }
    
    /**
     * @brief Load checkpoint from file
     * @param filepath Path to input file
     * @return Loaded checkpoint
     */
    static ModelCheckpoint load(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("Failed to open checkpoint file: " + filepath);
        }
        
        auto size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<uint8_t> data(size);
        file.read(reinterpret_cast<char*>(data.data()), size);
        
        return deserialize(data);
    }
    
    /**
     * @brief Validate checkpoint consistency
     */
    bool validate() const {
        if (spike_mean.size() != channel_config.num_channels) return false;
        if (spike_std.size() != channel_config.num_channels) return false;
        if (kalman_H.size() != channel_config.num_channels * 4) return false;
        
        if (has_pca) {
            if (pca_mean.size() != pca_n_features) return false;
            if (pca_components.size() != pca_n_features * pca_n_components) return false;
            if (kalman_H_latent.size() != pca_n_components * 4) return false;
        }
        
        return true;
    }
};

/**
 * @brief Create checkpoint from a trained KalmanDecoder
 * 
 * Extracts all necessary state for later restoration.
 */
ModelCheckpoint create_checkpoint(const KalmanDecoder& decoder);

/**
 * @brief Restore a KalmanDecoder from a checkpoint
 * 
 * Rebuilds the decoder with all trained parameters.
 */
void restore_from_checkpoint(KalmanDecoder& decoder, const ModelCheckpoint& checkpoint);

} // namespace phantomcore
