#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <memory>
#include <optional>

namespace phantomcore {

/**
 * @brief Ridge Regression (L2-regularized least squares)
 * 
 * Solves: min ||Xw - y||² + λ||w||²
 * 
 * Closed-form solution: w = (X^T X + λI)^-1 X^T y
 * 
 * Critical for BCI calibration because:
 * - Short calibration sessions → few samples → overfitting risk
 * - High-dimensional neural data → collinear channels
 * - Ridge prevents exploding weights and improves generalization
 * 
 * λ selection:
 * - Too small: overfits (weights explode, poor generalization)
 * - Too large: underfits (weights shrink to zero, loses signal)
 * - Optimal: use cross-validation or Bayesian optimization
 */
class RidgeRegression {
public:
    struct Config {
        float lambda = 1.0f;              // Regularization strength
        bool fit_intercept = true;        // Learn bias term
        bool normalize = false;           // Normalize features before fitting
        
        Config() = default;
    };
    
    struct CVResult {
        float best_lambda = 0.0f;
        float best_score = 0.0f;          // R² score
        std::vector<float> lambdas_tested;
        std::vector<float> cv_scores;
    };
    
    explicit RidgeRegression(const Config& config);
    RidgeRegression();
    ~RidgeRegression();
    
    RidgeRegression(RidgeRegression&&) noexcept;
    RidgeRegression& operator=(RidgeRegression&&) noexcept;
    
    /**
     * @brief Fit ridge regression
     * @param X Feature matrix [n_samples x n_features]
     * @param y Target matrix [n_samples x n_targets] or vector
     * @return True if fitting succeeded
     */
    bool fit(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y);
    bool fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);
    
    /**
     * @brief Predict targets for new data
     */
    Eigen::MatrixXf predict(const Eigen::MatrixXf& X) const;
    Eigen::VectorXf predict_single(const Eigen::VectorXf& x) const;
    
    /**
     * @brief Cross-validate to find optimal lambda
     * @param X Feature matrix
     * @param y Target matrix
     * @param lambdas Regularization values to try
     * @param n_folds Number of CV folds
     * @return Best lambda and CV scores
     */
    CVResult cross_validate(
        const Eigen::MatrixXf& X,
        const Eigen::MatrixXf& y,
        const std::vector<float>& lambdas = {0.001f, 0.01f, 0.1f, 1.0f, 10.0f, 100.0f},
        size_t n_folds = 5
    );
    
    /**
     * @brief Compute R² score
     */
    float score(const Eigen::MatrixXf& X, const Eigen::MatrixXf& y) const;
    
    // Accessors
    bool is_fitted() const { return fitted_; }
    const Eigen::MatrixXf& coefficients() const { return weights_; }
    const Eigen::VectorXf& intercept() const { return intercept_; }
    
    void set_lambda(float lambda) { config_.lambda = lambda; }
    
private:
    Config config_;
    Eigen::MatrixXf weights_;    // [n_features x n_targets]
    Eigen::VectorXf intercept_;  // [n_targets]
    Eigen::VectorXf X_mean_;     // For normalization
    Eigen::VectorXf X_std_;
    Eigen::VectorXf y_mean_;
    bool fitted_ = false;
};

/**
 * @brief Elastic Net Regression (L1 + L2 regularization)
 * 
 * Solves: min ||Xw - y||² + λ₁||w||₁ + λ₂||w||²
 * 
 * Combines benefits of:
 * - Ridge (L2): handles collinearity
 * - Lasso (L1): feature selection (sparse weights)
 * 
 * Useful when some neural channels are irrelevant.
 */
class ElasticNet {
public:
    struct Config {
        float alpha = 1.0f;       // Total regularization strength
        float l1_ratio = 0.5f;    // 0 = Ridge, 1 = Lasso, 0.5 = balanced
        size_t max_iterations = 1000;
        float tolerance = 1e-4f;
        bool fit_intercept = true;
        
        Config() = default;
    };
    
    explicit ElasticNet(const Config& config);
    ElasticNet();
    ~ElasticNet();
    
    bool fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);
    Eigen::VectorXf predict(const Eigen::MatrixXf& X) const;
    float predict_single(const Eigen::VectorXf& x) const;
    
    bool is_fitted() const { return fitted_; }
    const Eigen::VectorXf& coefficients() const { return weights_; }
    float intercept() const { return intercept_; }
    
    /// Number of non-zero coefficients (sparsity measure)
    size_t n_nonzero() const;
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    Eigen::VectorXf weights_;
    float intercept_ = 0.0f;
    bool fitted_ = false;
};

/**
 * @brief Bayesian Ridge Regression
 * 
 * Automatic regularization strength via Bayesian inference.
 * Learns the optimal λ from the data using empirical Bayes (Type-II ML).
 * 
 * Advantages:
 * - No hyperparameter tuning required
 * - Provides uncertainty estimates
 * - Robust to overfitting
 */
class BayesianRidge {
public:
    struct Config {
        size_t max_iterations = 300;
        float tolerance = 1e-3f;
        float alpha_init = 1.0f;   // Prior precision on weights
        float lambda_init = 1.0f;  // Prior precision on noise
        
        Config() = default;
    };
    
    explicit BayesianRidge(const Config& config);
    BayesianRidge();
    ~BayesianRidge();
    
    BayesianRidge(BayesianRidge&&) noexcept;
    BayesianRidge& operator=(BayesianRidge&&) noexcept;
    
    bool fit(const Eigen::MatrixXf& X, const Eigen::VectorXf& y);
    
    Eigen::VectorXf predict(const Eigen::MatrixXf& X) const;
    float predict_single(const Eigen::VectorXf& x) const;
    
    /**
     * @brief Predict with uncertainty
     * @return pair<mean, std> predictions
     */
    std::pair<Eigen::VectorXf, Eigen::VectorXf> predict_with_std(
        const Eigen::MatrixXf& X
    ) const;
    
    bool is_fitted() const { return fitted_; }
    const Eigen::VectorXf& coefficients() const { return weights_; }
    float intercept() const { return intercept_; }
    float alpha() const { return alpha_; }   // Learned weight precision
    float lambda() const { return lambda_; } // Learned noise precision
    
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    Config config_;
    Eigen::VectorXf weights_;
    Eigen::MatrixXf covariance_;  // Posterior covariance
    float intercept_ = 0.0f;
    float alpha_ = 1.0f;
    float lambda_ = 1.0f;
    bool fitted_ = false;
};

} // namespace phantomcore
