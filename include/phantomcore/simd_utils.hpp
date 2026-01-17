#pragma once

// Include SIMD headers FIRST before any other includes
#ifdef PHANTOMCORE_SIMD_ENABLED
    #if defined(_MSC_VER)
        #include <intrin.h>
    #endif
    #if defined(__AVX2__) || (defined(_MSC_VER) && defined(__AVX2__))
        #define PHANTOMCORE_HAS_AVX2 1
        #include <immintrin.h>
    #elif defined(__ARM_NEON)
        #define PHANTOMCORE_HAS_NEON 1
        #include <arm_neon.h>
    #endif
#endif

#include "types.hpp"
#include "aligned_allocator.hpp"
#include <array>
#include <cstddef>
#include <cstdint>

namespace phantomcore {
namespace simd {

// ============================================================================
// SIMD Detection and Configuration
// ============================================================================

/// Returns true if AVX2 is available at runtime
bool has_avx2();

/// Returns true if NEON is available at runtime
bool has_neon();

/// Returns a string describing the SIMD capabilities
const char* simd_info();

// ============================================================================
// Alignment Utilities
// ============================================================================

/**
 * @brief Check if pointer is 32-byte aligned (required for AVX2 aligned loads)
 */
[[nodiscard]] inline bool is_avx2_aligned(const void* ptr) noexcept {
    return aligned::is_aligned(ptr, aligned::AVX2_ALIGNMENT);
}

/**
 * @brief Assert AVX2 alignment in debug builds
 * In release builds, this is a no-op.
 */
inline void assert_avx2_aligned(const void* ptr, const char* context = nullptr) {
    aligned::assert_aligned(ptr, aligned::AVX2_ALIGNMENT, context);
}

// ============================================================================
// Vectorized Operations (Unaligned - Safe for any pointer)
// ============================================================================

/**
 * @brief Computes sum of float array using SIMD
 * @param data Pointer to float array (unaligned OK, aligned preferred)
 * @param size Number of elements
 * @return Sum of all elements
 * 
 * @note Uses _mm256_loadu_ps (unaligned load). For maximum performance
 *       with aligned data, use vector_sum_aligned().
 */
float vector_sum(const float* data, size_t size);

/**
 * @brief Computes mean of float array using SIMD
 */
float vector_mean(const float* data, size_t size);

/**
 * @brief Computes variance of float array using SIMD
 */
float vector_variance(const float* data, size_t size);

/**
 * @brief Computes standard deviation of float array using SIMD
 */
float vector_std(const float* data, size_t size);

/**
 * @brief Finds maximum value in float array using SIMD
 */
float vector_max(const float* data, size_t size);

/**
 * @brief Finds minimum value in float array using SIMD
 */
float vector_min(const float* data, size_t size);

/**
 * @brief Element-wise addition: result[i] = a[i] + b[i]
 */
void vector_add(const float* a, const float* b, float* result, size_t size);

/**
 * @brief Element-wise subtraction: result[i] = a[i] - b[i]
 */
void vector_sub(const float* a, const float* b, float* result, size_t size);

/**
 * @brief Element-wise multiplication: result[i] = a[i] * b[i]
 */
void vector_mul(const float* a, const float* b, float* result, size_t size);

/**
 * @brief Scalar multiplication: result[i] = a[i] * scalar
 */
void vector_scale(const float* a, float scalar, float* result, size_t size);

/**
 * @brief Dot product of two vectors
 */
float vector_dot(const float* a, const float* b, size_t size);

/**
 * @brief Fused multiply-add: result[i] = a[i] * b[i] + c[i]
 */
void vector_fma(const float* a, const float* b, const float* c, float* result, size_t size);

// ============================================================================
// Aligned Variants (Faster, requires 32-byte aligned pointers)
// ============================================================================

#ifdef PHANTOMCORE_HAS_AVX2

/**
 * @brief Sum using aligned loads (faster, REQUIRES 32-byte alignment)
 * @warning WILL CRASH (SegFault) if data is not 32-byte aligned!
 *          Use vector_sum() if alignment is not guaranteed.
 */
float vector_sum_aligned(const float* data, size_t size);

/**
 * @brief Dot product using aligned loads (faster, REQUIRES alignment)
 * @warning Both a and b MUST be 32-byte aligned!
 */
float vector_dot_aligned(const float* a, const float* b, size_t size);

/**
 * @brief Safe aligned operations using AlignedVector
 * These overloads guarantee alignment at compile time.
 */
inline float vector_sum(const AlignedVector<float>& data) {
    return vector_sum_aligned(data.data(), data.size());
}

inline float vector_dot(const AlignedVector<float>& a, const AlignedVector<float>& b) {
    return vector_dot_aligned(a.data(), b.data(), std::min(a.size(), b.size()));
}

/**
 * @brief Safe aligned operations using AlignedBuffer
 */
template<size_t N>
inline float vector_sum(const AlignedBuffer<float, N>& data) {
    return vector_sum_aligned(data.data(), N);
}

template<size_t N>
inline float vector_dot(const AlignedBuffer<float, N>& a, const AlignedBuffer<float, N>& b) {
    return vector_dot_aligned(a.data(), b.data(), N);
}

#endif // PHANTOMCORE_HAS_AVX2

// ============================================================================
// Neural Signal Processing Operations
// ============================================================================

/**
 * @brief Applies threshold detection across all channels
 * @param data Input signal array
 * @param thresholds Per-channel thresholds
 * @param crossings Output: 1 if crossed threshold, 0 otherwise
 * @param size Number of channels
 */
void threshold_crossing(
    const float* data,
    const float* thresholds,
    int32_t* crossings,
    size_t size
);

/**
 * @brief Computes per-channel z-scores: (x - mean) / std
 * @param data Input signal array
 * @param means Per-channel means
 * @param stds Per-channel standard deviations
 * @param result Output z-scores
 * @param size Number of channels
 */
void compute_zscores(
    const float* data,
    const float* means,
    const float* stds,
    float* result,
    size_t size
);

/**
 * @brief Applies exponential smoothing filter
 * @param current Current values
 * @param previous Previous smoothed values
 * @param alpha Smoothing factor (0-1)
 * @param result Output smoothed values
 * @param size Number of elements
 */
void exponential_smooth(
    const float* current,
    const float* previous,
    float alpha,
    float* result,
    size_t size
);

/**
 * @brief Matrix-vector multiplication for decoder weights
 * @param matrix Weight matrix (rows x cols)
 * @param vector Input vector (cols)
 * @param result Output vector (rows)
 * @param rows Number of output dimensions
 * @param cols Number of input dimensions
 */
void matrix_vector_mul(
    const float* matrix,
    const float* vector,
    float* result,
    size_t rows,
    size_t cols
);

// ============================================================================
// Optimized Operations for Neural Data
// ============================================================================

/**
 * @brief Processes a full 142-channel spike count array
 * Specialized for the exact channel count with optimal SIMD utilization
 */
struct ChannelProcessor {
    /// Computes mean firing rate across all channels
    static float compute_mean_rate(const AlignedSpikeData& spikes);
    
    /// Computes per-channel z-scores
    static void compute_zscores(
        const AlignedSpikeData& spikes,
        const AlignedSpikeData& means,
        const AlignedSpikeData& stds,
        AlignedSpikeData& result
    );
    
    /// Applies linear decoder weights
    static Vec2 apply_decoder(
        const AlignedSpikeData& spikes,
        const std::array<float, NUM_CHANNELS>& weights_x,
        const std::array<float, NUM_CHANNELS>& weights_y,
        float bias_x,
        float bias_y
    );
    
    /// Updates running mean/variance with new sample
    static void update_statistics(
        const AlignedSpikeData& spikes,
        AlignedSpikeData& running_mean,
        AlignedSpikeData& running_var,
        size_t sample_count
    );
};

}  // namespace simd
}  // namespace phantomcore
