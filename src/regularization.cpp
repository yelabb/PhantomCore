#include "phantomcore/regularization.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace phantomcore {

// ============================================================================
// RidgeRegression Implementation
// ============================================================================

RidgeRegression::RidgeRegression(const Config& config) : config_(config) {}
RidgeRegression::~RidgeRegression() = default;
RidgeRegression::RidgeRegression(RidgeRegression&&) noexcept = default;
RidgeRegression& RidgeRegression::operator=(RidgeRegression&&) noexcept = default;

bool RidgeRegression::fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
    return fit(X, y.replicate(1, 1));  // Convert to matrix
}

bool RidgeRegression::fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y) {
    if (X.rows() != y.rows() || X.rows() < 1) {
        return false;
    }
    
    const Eigen::Index n_samples = X.rows();
    const Eigen::Index n_features = X.cols();
    const Eigen::Index n_targets = y.cols();
    
    Eigen::MatrixXf X_proc = X;
    Eigen::MatrixXf y_proc = y;
    
    // Center the data
    if (config_.fit_intercept) {
        X_mean_ = X.colwise().mean();
        y_mean_ = y.colwise().mean();
        X_proc = X.rowwise() - X_mean_.transpose();
        y_proc = y.rowwise() - y_mean_.transpose();
    }
    
    // Optionally normalize
    if (config_.normalize) {
        X_std_ = Eigen::VectorXf(n_features);
        for (Eigen::Index i = 0; i < n_features; ++i) {
            float var = X_proc.col(i).squaredNorm() / static_cast<float>(n_samples);
            X_std_(i) = std::sqrt(var);
            if (X_std_(i) > 1e-10f) {
                X_proc.col(i) /= X_std_(i);
            }
        }
    }
    
    // Ridge solution: w = (X^T X + λI)^-1 X^T y
    Eigen::MatrixXf XtX = X_proc.transpose() * X_proc;
    XtX += config_.lambda * Eigen::MatrixXf::Identity(n_features, n_features);
    
    Eigen::MatrixXf Xty = X_proc.transpose() * y_proc;
    
    // Solve using Cholesky decomposition (more stable than direct inverse)
    Eigen::LLT<Eigen::MatrixXf> llt(XtX);
    if (llt.info() != Eigen::Success) {
        // Fall back to pseudoinverse
        weights_ = XtX.completeOrthogonalDecomposition().solve(Xty);
    } else {
        weights_ = llt.solve(Xty);
    }
    
    // Undo normalization on weights
    if (config_.normalize && X_std_.size() > 0) {
        for (Eigen::Index i = 0; i < n_features; ++i) {
            if (X_std_(i) > 1e-10f) {
                weights_.row(i) /= X_std_(i);
            }
        }
    }
    
    // Compute intercept
    if (config_.fit_intercept) {
        intercept_ = y_mean_ - (X_mean_.transpose() * weights_).transpose();
    } else {
        intercept_ = Eigen::VectorXf::Zero(n_targets);
    }
    
    fitted_ = true;
    return true;
}

Eigen::MatrixXf RidgeRegression::predict(const Eigen::MatrixXf& X) const {
    if (!fitted_) {
        return Eigen::MatrixXf();
    }
    
    Eigen::MatrixXf y_pred = X * weights_;
    if (config_.fit_intercept) {
        y_pred.rowwise() += intercept_.transpose();
    }
    return y_pred;
}

Eigen::VectorXf RidgeRegression::predict_single(const Eigen::VectorXf& x) const {
    if (!fitted_) {
        return Eigen::VectorXf();
    }
    
    Eigen::VectorXf y_pred = weights_.transpose() * x;
    if (config_.fit_intercept) {
        y_pred += intercept_;
    }
    return y_pred;
}

RidgeRegression::CVResult RidgeRegression::cross_validate(
    const Eigen::MatrixXf& X,
    const Eigen::MatrixXf& y,
    const std::vector<float>& lambdas,
    size_t n_folds
) {
    CVResult result;
    result.lambdas_tested = lambdas;
    result.cv_scores.resize(lambdas.size(), 0.0f);
    result.best_score = -std::numeric_limits<float>::infinity();
    
    const Eigen::Index n_samples = X.rows();
    const Eigen::Index fold_size = n_samples / static_cast<Eigen::Index>(n_folds);
    
    // Create shuffled indices
    std::vector<Eigen::Index> indices(static_cast<size_t>(n_samples));
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);
    
    for (size_t li = 0; li < lambdas.size(); ++li) {
        float lambda = lambdas[li];
        float total_score = 0.0f;
        
        for (size_t fold = 0; fold < n_folds; ++fold) {
            // Split into train/test
            Eigen::Index test_start = static_cast<Eigen::Index>(fold) * fold_size;
            Eigen::Index test_end = (fold == n_folds - 1) ? n_samples : test_start + fold_size;
            Eigen::Index test_size = test_end - test_start;
            Eigen::Index train_size = n_samples - test_size;
            
            Eigen::MatrixXf X_train(train_size, X.cols());
            Eigen::MatrixXf y_train(train_size, y.cols());
            Eigen::MatrixXf X_test(test_size, X.cols());
            Eigen::MatrixXf y_test(test_size, y.cols());
            
            Eigen::Index train_idx = 0, test_idx = 0;
            for (Eigen::Index i = 0; i < n_samples; ++i) {
                if (i >= test_start && i < test_end) {
                    X_test.row(test_idx) = X.row(indices[static_cast<size_t>(i)]);
                    y_test.row(test_idx) = y.row(indices[static_cast<size_t>(i)]);
                    test_idx++;
                } else {
                    X_train.row(train_idx) = X.row(indices[static_cast<size_t>(i)]);
                    y_train.row(train_idx) = y.row(indices[static_cast<size_t>(i)]);
                    train_idx++;
                }
            }
            
            // Fit and score
            RidgeRegression fold_model(config_);
            fold_model.set_lambda(lambda);
            if (fold_model.fit(X_train, y_train)) {
                total_score += fold_model.score(X_test, y_test);
            }
        }
        
        result.cv_scores[li] = total_score / static_cast<float>(n_folds);
        
        if (result.cv_scores[li] > result.best_score) {
            result.best_score = result.cv_scores[li];
            result.best_lambda = lambda;
        }
    }
    
    return result;
}

float RidgeRegression::score(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y) const {
    if (!fitted_) return 0.0f;
    
    Eigen::MatrixXf y_pred = predict(X);
    
    // R² = 1 - SS_res / SS_tot
    float ss_res = (y - y_pred).squaredNorm();
    Eigen::VectorXf y_mean = y.colwise().mean();
    float ss_tot = (y.rowwise() - y_mean.transpose()).squaredNorm();
    
    if (ss_tot < 1e-10f) return 0.0f;
    return 1.0f - ss_res / ss_tot;
}

// ============================================================================
// ElasticNet Implementation (Coordinate Descent)
// ============================================================================

struct ElasticNet::Impl {
    // Soft thresholding operator for L1
    static float soft_threshold(float x, float lambda) {
        if (x > lambda) return x - lambda;
        if (x < -lambda) return x + lambda;
        return 0.0f;
    }
};

ElasticNet::ElasticNet(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {}

ElasticNet::~ElasticNet() = default;

bool ElasticNet::fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
    const Eigen::Index n_samples = X.rows();
    const Eigen::Index n_features = X.cols();
    
    // Center data
    Eigen::VectorXf X_mean = X.colwise().mean();
    float y_mean = y.mean();
    
    Eigen::MatrixXf X_c = X.rowwise() - X_mean.transpose();
    Eigen::VectorXf y_c = y.array() - y_mean;
    
    // Precompute X^T X diagonal and X^T y
    Eigen::VectorXf X_sq = X_c.colwise().squaredNorm().transpose();
    Eigen::VectorXf Xty = X_c.transpose() * y_c;
    
    // Initialize weights
    weights_ = Eigen::VectorXf::Zero(n_features);
    
    float l1_reg = config_.alpha * config_.l1_ratio;
    float l2_reg = config_.alpha * (1.0f - config_.l1_ratio);
    
    // Coordinate descent
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        Eigen::VectorXf weights_old = weights_;
        
        for (Eigen::Index j = 0; j < n_features; ++j) {
            // Compute residual without feature j
            Eigen::VectorXf residual = y_c - X_c * weights_;
            residual += X_c.col(j) * weights_(j);
            
            // Partial correlation
            float rho = X_c.col(j).dot(residual);
            
            // Elastic net update with soft thresholding
            float denom = X_sq(j) + l2_reg;
            if (denom > 1e-10f) {
                weights_(j) = Impl::soft_threshold(rho, l1_reg * n_samples) / denom;
            }
        }
        
        // Check convergence
        float delta = (weights_ - weights_old).norm();
        if (delta < config_.tolerance) {
            break;
        }
    }
    
    // Compute intercept
    if (config_.fit_intercept) {
        intercept_ = y_mean - X_mean.dot(weights_);
    }
    
    fitted_ = true;
    return true;
}

Eigen::VectorXf ElasticNet::predict(const Eigen::MatrixXf& X) const {
    if (!fitted_) return Eigen::VectorXf();
    
    Eigen::VectorXf y_pred = X * weights_;
    if (config_.fit_intercept) {
        y_pred.array() += intercept_;
    }
    return y_pred;
}

float ElasticNet::predict_single(const Eigen::VectorXf& x) const {
    if (!fitted_) return 0.0f;
    float y = weights_.dot(x);
    if (config_.fit_intercept) y += intercept_;
    return y;
}

size_t ElasticNet::n_nonzero() const {
    return static_cast<size_t>((weights_.array().abs() > 1e-10f).count());
}

// ============================================================================
// BayesianRidge Implementation (Empirical Bayes / Type-II ML)
// ============================================================================

struct BayesianRidge::Impl {
    Eigen::MatrixXf posterior_cov;
};

BayesianRidge::BayesianRidge(const Config& config)
    : impl_(std::make_unique<Impl>()), config_(config) {}

BayesianRidge::~BayesianRidge() = default;
BayesianRidge::BayesianRidge(BayesianRidge&&) noexcept = default;
BayesianRidge& BayesianRidge::operator=(BayesianRidge&&) noexcept = default;

bool BayesianRidge::fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y) {
    const Eigen::Index n_samples = X.rows();
    const Eigen::Index n_features = X.cols();
    
    // Center data
    Eigen::VectorXf X_mean = X.colwise().mean();
    float y_mean = y.mean();
    
    Eigen::MatrixXf X_c = X.rowwise() - X_mean.transpose();
    Eigen::VectorXf y_c = y.array() - y_mean;
    
    // SVD for numerical stability
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(X_c, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf s = svd.singularValues();
    Eigen::VectorXf s_sq = s.array().square();
    
    // Transform to eigenspace
    Eigen::VectorXf Uty = svd.matrixU().transpose() * y_c;
    
    alpha_ = config_.alpha_init;
    lambda_ = config_.lambda_init;
    
    // EM iterations
    for (size_t iter = 0; iter < config_.max_iterations; ++iter) {
        float alpha_old = alpha_;
        float lambda_old = lambda_;
        
        // Compute eigenvalues of posterior precision
        Eigen::VectorXf d = s_sq.array() + alpha_ / lambda_;
        
        // Posterior mean in eigenspace
        Eigen::VectorXf gamma = s_sq.array() / d.array();
        
        // Compute weights
        Eigen::VectorXf w_eigen = (s.array() / d.array()) * Uty.array();
        weights_ = svd.matrixV() * w_eigen;
        
        // Update alpha (precision on weights)
        float gamma_sum = gamma.sum();
        float w_sq = weights_.squaredNorm();
        alpha_ = gamma_sum / std::max(w_sq, 1e-10f);
        
        // Update lambda (precision on noise)
        Eigen::VectorXf residual = y_c - X_c * weights_;
        float sse = residual.squaredNorm();
        lambda_ = (static_cast<float>(n_samples) - gamma_sum) / std::max(sse, 1e-10f);
        
        // Check convergence
        if (std::abs(alpha_ - alpha_old) < config_.tolerance * alpha_old &&
            std::abs(lambda_ - lambda_old) < config_.tolerance * lambda_old) {
            break;
        }
    }
    
    // Compute posterior covariance for uncertainty
    Eigen::VectorXf d = s_sq.array() + alpha_ / lambda_;
    Eigen::MatrixXf V = svd.matrixV();
    covariance_ = V * (d.array().inverse().matrix().asDiagonal()) * V.transpose() / lambda_;
    
    // Compute intercept
    intercept_ = y_mean - X_mean.dot(weights_);
    
    fitted_ = true;
    return true;
}

Eigen::VectorXf BayesianRidge::predict(const Eigen::MatrixXf& X) const {
    if (!fitted_) return Eigen::VectorXf();
    return (X * weights_).array() + intercept_;
}

float BayesianRidge::predict_single(const Eigen::VectorXf& x) const {
    if (!fitted_) return 0.0f;
    return weights_.dot(x) + intercept_;
}

std::pair<Eigen::VectorXf, Eigen::VectorXf> BayesianRidge::predict_with_std(
    const Eigen::MatrixXf& X
) const {
    Eigen::VectorXf mean = predict(X);
    
    // Predictive variance: σ² = 1/λ + x^T Σ x
    Eigen::VectorXf std(X.rows());
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        float var = 1.0f / lambda_ + (X.row(i) * covariance_ * X.row(i).transpose())(0);
        std(i) = std::sqrt(std::max(var, 0.0f));
    }
    
    return {mean, std};
}

} // namespace phantomcore
