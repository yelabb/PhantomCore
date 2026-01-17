#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <memory>
#include <optional>

namespace phantomcore {

/**
 * @brief Principal Component Analysis for Neural Dimensionality Reduction
 * 
 * In BCI applications, raw neural channels (142 in our case) are highly
 * correlated and noisy. PCA projects onto a lower-dimensional manifold:
 * 
 *   142 channels → 10-20 latent dimensions
 * 
 * Benefits:
 * - Noise reduction: First PCs capture signal, later PCs capture noise
 * - Computational efficiency: Decode from 15 dims instead of 142
 * - Regularization: Implicit dimensionality constraint
 * - Interpretability: Latent factors often correspond to movement primitives
 * 
 * Mathematical formulation:
 *   X_centered = X - μ
 *   X_centered = U * Σ * V^T  (SVD)
 *   Z = X_centered * V[:, :k]  (projection to k dimensions)
 */
class PCAProjector {
public:
    struct Config {
        size_t n_components = 15;           // Target latent dimensions
        float variance_threshold = 0.95f;   // Or use variance explained
        bool use_variance_threshold = false; // If true, ignore n_components
        bool center = true;                  // Subtract mean
        bool scale = false;                  // Divide by std (standardize)
    };
    
    explicit PCAProjector(const Config& config = {});
    ~PCAProjector();
    
    // Move-only
    PCAProjector(const PCAProjector&) = delete;
    PCAProjector& operator=(const PCAProjector&) = delete;
    PCAProjector(PCAProjector&&) noexcept;
    PCAProjector& operator=(PCAProjector&&) noexcept;
    
    /**
     * @brief Fit PCA on training data
     * @param data Matrix of shape [n_samples x n_features]
     * @return True if fitting succeeded
     */
    bool fit(const Eigen::MatrixXf& data);
    
    /**
     * @brief Transform new data to latent space
     * @param data Matrix [n_samples x n_features] or vector [n_features]
     * @return Projected data [n_samples x n_components]
     */
    Eigen::MatrixXf transform(const Eigen::MatrixXf& data) const;
    Eigen::VectorXf transform(const Eigen::VectorXf& sample) const;
    
    /**
     * @brief Transform aligned spike data (optimized for real-time)
     * @param spikes 142-channel spike counts
     * @return Latent representation
     */
    Eigen::VectorXf transform(const AlignedSpikeData& spikes) const;
    
    /**
     * @brief Inverse transform from latent to original space
     */
    Eigen::MatrixXf inverse_transform(const Eigen::MatrixXf& latent) const;
    
    /**
     * @brief Fit and transform in one step
     */
    Eigen::MatrixXf fit_transform(const Eigen::MatrixXf& data);
    
    // Accessors
    bool is_fitted() const { return fitted_; }
    size_t n_components() const { return n_components_; }
    size_t n_features() const { return n_features_; }
    
    /// Get explained variance ratio for each component
    Eigen::VectorXf explained_variance_ratio() const;
    
    /// Get cumulative explained variance
    float cumulative_variance_explained() const;
    
    /// Get principal components (loadings) [n_features x n_components]
    const Eigen::MatrixXf& components() const;
    
    /// Get feature means
    const Eigen::VectorXf& mean() const;
    
    // ========================================================================
    // Serialization (for ModelCheckpoint)
    // ========================================================================
    
    /**
     * @brief Get all state for serialization
     */
    struct SerializedState {
        size_t n_features = 0;
        size_t n_components = 0;
        std::vector<float> mean;           // [n_features]
        std::vector<float> std;            // [n_features] (if scaled)
        std::vector<float> components;     // [n_features x n_components] row-major
        std::vector<float> explained_var;  // [n_components]
        float total_variance_explained = 0.0f;
        bool scaled = false;
    };
    
    SerializedState get_serialized_state() const;
    
    /**
     * @brief Restore PCA state from serialized data
     */
    void load_serialized_state(const SerializedState& state);
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    bool fitted_ = false;
    size_t n_components_ = 0;
    size_t n_features_ = 0;
};

/**
 * @brief Factor Analysis for Neural Dimensionality Reduction
 * 
 * FA is often preferred over PCA in neuroscience because:
 * - Explicitly models observation noise separate from latent factors
 * - Better handles heteroscedastic noise across channels
 * - More interpretable latent factors
 * 
 * Model: X = μ + L * z + ε
 *   where z ~ N(0, I), ε ~ N(0, Ψ) with diagonal Ψ
 * 
 * Note: More computationally expensive than PCA.
 */
class FactorAnalysis {
public:
    struct Config {
        size_t n_factors = 15;
        size_t max_iterations = 100;
        float tolerance = 1e-4f;
    };
    
    explicit FactorAnalysis(const Config& config = {});
    ~FactorAnalysis();
    
    FactorAnalysis(FactorAnalysis&&) noexcept;
    FactorAnalysis& operator=(FactorAnalysis&&) noexcept;
    
    bool fit(const Eigen::MatrixXf& data);
    Eigen::MatrixXf transform(const Eigen::MatrixXf& data) const;
    Eigen::VectorXf transform(const Eigen::VectorXf& sample) const;
    
    bool is_fitted() const { return fitted_; }
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    bool fitted_ = false;
};

} // namespace phantomcore
