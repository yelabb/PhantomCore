#include "phantomcore/dimensionality_reduction.hpp"
#include <algorithm>
#include <cmath>

namespace phantomcore {

// ============================================================================
// PCAProjector Implementation
// ============================================================================

struct PCAProjector::Impl {
    Eigen::MatrixXf components;      // [n_features x n_components] - principal axes
    Eigen::VectorXf mean;            // [n_features] - feature means
    Eigen::VectorXf std;             // [n_features] - feature std (if scaling)
    Eigen::VectorXf singular_values; // Singular values
    Eigen::VectorXf explained_var;   // Variance explained by each component
    float total_variance = 0.0f;
};

PCAProjector::PCAProjector(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {}

PCAProjector::~PCAProjector() = default;
PCAProjector::PCAProjector(PCAProjector&&) noexcept = default;
PCAProjector& PCAProjector::operator=(PCAProjector&&) noexcept = default;

bool PCAProjector::fit(const Eigen::MatrixXf& data) {
    if (data.rows() < 2 || data.cols() < 1) {
        return false;
    }
    
    const Eigen::Index n_samples = data.rows();
    const Eigen::Index n_features = data.cols();
    n_features_ = static_cast<size_t>(n_features);
    
    // Compute mean
    impl_->mean = data.colwise().mean();
    
    // Center the data
    Eigen::MatrixXf centered = data.rowwise() - impl_->mean.transpose();
    
    // Optional: standardize (divide by std)
    if (config_.scale) {
        impl_->std = Eigen::VectorXf(n_features);
        for (Eigen::Index i = 0; i < n_features; ++i) {
            float variance = centered.col(i).squaredNorm() / static_cast<float>(n_samples - 1);
            impl_->std(i) = std::sqrt(variance);
            if (impl_->std(i) > 1e-10f) {
                centered.col(i) /= impl_->std(i);
            }
        }
    }
    
    // SVD: X = U * Σ * V^T
    // The right singular vectors V are the principal components
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeThinV);
    
    impl_->singular_values = svd.singularValues();
    
    // Compute explained variance
    impl_->explained_var = impl_->singular_values.array().square() / 
                           static_cast<float>(n_samples - 1);
    impl_->total_variance = impl_->explained_var.sum();
    
    // Determine number of components
    if (config_.use_variance_threshold) {
        // Find k such that cumulative variance >= threshold
        float cumsum = 0.0f;
        n_components_ = 0;
        for (Eigen::Index i = 0; i < impl_->explained_var.size(); ++i) {
            cumsum += impl_->explained_var(i);
            n_components_++;
            if (cumsum / impl_->total_variance >= config_.variance_threshold) {
                break;
            }
        }
    } else {
        n_components_ = std::min(config_.n_components, 
                                  static_cast<size_t>(svd.matrixV().cols()));
    }
    
    // Store principal components (first k columns of V)
    impl_->components = svd.matrixV().leftCols(static_cast<Eigen::Index>(n_components_));
    
    fitted_ = true;
    return true;
}

Eigen::MatrixXf PCAProjector::transform(const Eigen::MatrixXf& data) const {
    if (!fitted_) {
        return Eigen::MatrixXf();
    }
    
    // Center (and optionally scale)
    Eigen::MatrixXf centered = data.rowwise() - impl_->mean.transpose();
    if (config_.scale && impl_->std.size() > 0) {
        for (Eigen::Index i = 0; i < centered.cols(); ++i) {
            if (impl_->std(i) > 1e-10f) {
                centered.col(i) /= impl_->std(i);
            }
        }
    }
    
    // Project: Z = X_centered * V
    return centered * impl_->components;
}

Eigen::VectorXf PCAProjector::transform(const Eigen::VectorXf& sample) const {
    if (!fitted_) {
        return Eigen::VectorXf();
    }
    
    Eigen::VectorXf centered = sample - impl_->mean;
    if (config_.scale && impl_->std.size() > 0) {
        for (Eigen::Index i = 0; i < centered.size(); ++i) {
            if (impl_->std(i) > 1e-10f) {
                centered(i) /= impl_->std(i);
            }
        }
    }
    
    return impl_->components.transpose() * centered;
}

Eigen::VectorXf PCAProjector::transform(const AlignedSpikeData& spikes) const {
    if (!fitted_) {
        return Eigen::VectorXf();
    }
    
    // Map aligned data to Eigen vector (no copy) - using legacy fixed size
    Eigen::Map<const Eigen::VectorXf> sample_map(spikes.data(), 
                                                  static_cast<Eigen::Index>(142));
    Eigen::VectorXf sample = sample_map;  // Explicit copy to resolve overload
    return transform(sample);
}

Eigen::MatrixXf PCAProjector::inverse_transform(const Eigen::MatrixXf& latent) const {
    if (!fitted_) {
        return Eigen::MatrixXf();
    }
    
    // Reconstruct: X = Z * V^T + μ
    Eigen::MatrixXf reconstructed = latent * impl_->components.transpose();
    
    if (config_.scale && impl_->std.size() > 0) {
        for (Eigen::Index i = 0; i < reconstructed.cols(); ++i) {
            reconstructed.col(i) *= impl_->std(i);
        }
    }
    
    return reconstructed.rowwise() + impl_->mean.transpose();
}

Eigen::MatrixXf PCAProjector::fit_transform(const Eigen::MatrixXf& data) {
    if (!fit(data)) {
        return Eigen::MatrixXf();
    }
    return transform(data);
}

Eigen::VectorXf PCAProjector::explained_variance_ratio() const {
    if (!fitted_ || impl_->total_variance < 1e-10f) {
        return Eigen::VectorXf();
    }
    return impl_->explained_var.head(static_cast<Eigen::Index>(n_components_)) / 
           impl_->total_variance;
}

float PCAProjector::cumulative_variance_explained() const {
    if (!fitted_ || impl_->total_variance < 1e-10f) {
        return 0.0f;
    }
    return impl_->explained_var.head(static_cast<Eigen::Index>(n_components_)).sum() / 
           impl_->total_variance;
}

const Eigen::MatrixXf& PCAProjector::components() const {
    return impl_->components;
}

const Eigen::VectorXf& PCAProjector::mean() const {
    return impl_->mean;
}

// ============================================================================
// FactorAnalysis Implementation (EM Algorithm)
// ============================================================================

struct FactorAnalysis::Impl {
    Eigen::MatrixXf loadings;    // [n_features x n_factors] - factor loadings L
    Eigen::VectorXf mean;        // [n_features]
    Eigen::VectorXf noise_var;   // [n_features] - diagonal of Ψ (uniquenesses)
    
    // For inference
    Eigen::MatrixXf W;           // Precomputed for transform
};

FactorAnalysis::FactorAnalysis(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {}

FactorAnalysis::~FactorAnalysis() = default;
FactorAnalysis::FactorAnalysis(FactorAnalysis&&) noexcept = default;
FactorAnalysis& FactorAnalysis::operator=(FactorAnalysis&&) noexcept = default;

bool FactorAnalysis::fit(const Eigen::MatrixXf& data) {
    // EM algorithm for Factor Analysis
    // Model: x = μ + L*z + ε, where z~N(0,I), ε~N(0,Ψ)
    
    const Eigen::Index n_samples = data.rows();
    const Eigen::Index n_features = data.cols();
    const Eigen::Index n_factors = static_cast<Eigen::Index>(config_.n_factors);
    
    if (n_samples < n_factors || n_features < n_factors) {
        return false;
    }
    
    // Initialize with PCA
    impl_->mean = data.colwise().mean();
    Eigen::MatrixXf centered = data.rowwise() - impl_->mean.transpose();
    
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
    
    // Initialize loadings from first k principal components
    Eigen::VectorXf sqrt_s = svd.singularValues().head(n_factors).array().sqrt() / 
                              std::sqrt(static_cast<float>(n_samples - 1));
    impl_->loadings = svd.matrixV().leftCols(n_factors) * sqrt_s.asDiagonal();
    
    // Initialize noise variance from residual
    impl_->noise_var = Eigen::VectorXf::Ones(n_features) * 0.1f;
    
    // EM iterations
    float prev_ll = -std::numeric_limits<float>::infinity();
    
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        // E-step: compute expected latent factors
        // E[z|x] = (I + L^T Ψ^-1 L)^-1 L^T Ψ^-1 (x - μ)
        
        Eigen::MatrixXf Psi_inv = impl_->noise_var.array().inverse().matrix().asDiagonal();
        Eigen::MatrixXf M = Eigen::MatrixXf::Identity(n_factors, n_factors) + 
                            impl_->loadings.transpose() * Psi_inv * impl_->loadings;
        Eigen::MatrixXf M_inv = M.inverse();
        
        // Expected factors for all samples
        Eigen::MatrixXf Ez = centered * Psi_inv * impl_->loadings * M_inv.transpose();
        
        // E[z z^T | x] = M^-1 + E[z|x] E[z|x]^T
        Eigen::MatrixXf Ezz_sum = static_cast<float>(n_samples) * M_inv;
        for (Eigen::Index i = 0; i < n_samples; ++i) {
            Ezz_sum += Ez.row(i).transpose() * Ez.row(i);
        }
        
        // M-step: update parameters
        // L_new = (Σ x E[z]^T) (Σ E[zz^T])^-1
        Eigen::MatrixXf L_new = centered.transpose() * Ez * Ezz_sum.inverse();
        
        // Ψ_new = diag(Σ (x - μ)(x - μ)^T - L_new Σ E[z] (x - μ)^T) / N
        Eigen::MatrixXf cov = centered.transpose() * centered / static_cast<float>(n_samples);
        Eigen::MatrixXf Ez_x = Ez.transpose() * centered / static_cast<float>(n_samples);
        Eigen::VectorXf psi_new = (cov - L_new * Ez_x).diagonal();
        psi_new = psi_new.array().max(1e-6f);  // Ensure positive
        
        impl_->loadings = L_new;
        impl_->noise_var = psi_new;
        
        // Check convergence (simplified: check parameter change)
        float ll = -0.5f * (cov.trace() - (impl_->loadings * Ez_x).trace());
        if (std::abs(ll - prev_ll) < config_.tolerance) {
            break;
        }
        prev_ll = ll;
    }
    
    // Precompute transform matrix: W = (L^T Ψ^-1 L + I)^-1 L^T Ψ^-1
    Eigen::MatrixXf Psi_inv = impl_->noise_var.array().inverse().matrix().asDiagonal();
    Eigen::MatrixXf M = Eigen::MatrixXf::Identity(n_factors, n_factors) + 
                        impl_->loadings.transpose() * Psi_inv * impl_->loadings;
    impl_->W = M.inverse() * impl_->loadings.transpose() * Psi_inv;
    
    fitted_ = true;
    return true;
}

Eigen::MatrixXf FactorAnalysis::transform(const Eigen::MatrixXf& data) const {
    if (!fitted_) return Eigen::MatrixXf();
    
    Eigen::MatrixXf centered = data.rowwise() - impl_->mean.transpose();
    return (impl_->W * centered.transpose()).transpose();
}

Eigen::VectorXf FactorAnalysis::transform(const Eigen::VectorXf& sample) const {
    if (!fitted_) return Eigen::VectorXf();
    
    return impl_->W * (sample - impl_->mean);
}

} // namespace phantomcore
