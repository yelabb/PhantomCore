#include "phantomcore/simd_utils.hpp"
#include <cmath>
#include <numeric>

namespace phantomcore {
namespace simd {

// ============================================================================
// SIMD Detection
// ============================================================================

bool has_avx2() {
#ifdef PHANTOMCORE_HAS_AVX2
    return true;
#else
    return false;
#endif
}

bool has_neon() {
#ifdef PHANTOMCORE_HAS_NEON
    return true;
#else
    return false;
#endif
}

const char* simd_info() {
#if defined(PHANTOMCORE_HAS_AVX2)
    return "AVX2 (256-bit)";
#elif defined(PHANTOMCORE_HAS_NEON)
    return "NEON (128-bit)";
#else
    return "Scalar (no SIMD)";
#endif
}

// ============================================================================
// Vectorized Operations - AVX2 Implementation
// ============================================================================

#ifdef PHANTOMCORE_HAS_AVX2

float vector_sum(const float* data, size_t size) {
    __m256 sum_vec = _mm256_setzero_ps();
    
    size_t i = 0;
    // Process 8 floats at a time
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        sum_vec = _mm256_add_ps(sum_vec, v);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remainder
    for (; i < size; ++i) {
        result += data[i];
    }
    
    return result;
}

float vector_mean(const float* data, size_t size) {
    return vector_sum(data, size) / static_cast<float>(size);
}

float vector_variance(const float* data, size_t size) {
    float mean = vector_mean(data, size);
    __m256 mean_vec = _mm256_set1_ps(mean);
    __m256 sum_sq = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        __m256 diff = _mm256_sub_ps(v, mean_vec);
        sum_sq = _mm256_fmadd_ps(diff, diff, sum_sq);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum_sq, 1);
    __m128 lo = _mm256_castps256_ps128(sum_sq);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remainder
    for (; i < size; ++i) {
        float diff = data[i] - mean;
        result += diff * diff;
    }
    
    return result / static_cast<float>(size);
}

float vector_std(const float* data, size_t size) {
    return std::sqrt(vector_variance(data, size));
}

float vector_max(const float* data, size_t size) {
    if (size == 0) return 0.0f;
    
    __m256 max_vec = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        max_vec = _mm256_max_ps(max_vec, v);
    }
    
    // Horizontal max
    __m128 hi = _mm256_extractf128_ps(max_vec, 1);
    __m128 lo = _mm256_castps256_ps128(max_vec);
    __m128 max128 = _mm_max_ps(hi, lo);
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
    max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
    float result = _mm_cvtss_f32(max128);
    
    for (; i < size; ++i) {
        result = std::max(result, data[i]);
    }
    
    return result;
}

float vector_min(const float* data, size_t size) {
    if (size == 0) return 0.0f;
    
    __m256 min_vec = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_loadu_ps(data + i);
        min_vec = _mm256_min_ps(min_vec, v);
    }
    
    // Horizontal min
    __m128 hi = _mm256_extractf128_ps(min_vec, 1);
    __m128 lo = _mm256_castps256_ps128(min_vec);
    __m128 min128 = _mm_min_ps(hi, lo);
    min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(2, 3, 0, 1)));
    min128 = _mm_min_ps(min128, _mm_shuffle_ps(min128, min128, _MM_SHUFFLE(1, 0, 3, 2)));
    float result = _mm_cvtss_f32(min128);
    
    for (; i < size; ++i) {
        result = std::min(result, data[i]);
    }
    
    return result;
}

void vector_add(const float* a, const float* b, float* result, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(result + i, _mm256_add_ps(va, vb));
    }
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void vector_sub(const float* a, const float* b, float* result, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(result + i, _mm256_sub_ps(va, vb));
    }
    for (; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void vector_mul(const float* a, const float* b, float* result, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(result + i, _mm256_mul_ps(va, vb));
    }
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void vector_scale(const float* a, float scalar, float* result, size_t size) {
    __m256 s = _mm256_set1_ps(scalar);
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(result + i, _mm256_mul_ps(va, s));
    }
    for (; i < size; ++i) {
        result[i] = a[i] * scalar;
    }
}

float vector_dot(const float* a, const float* b, size_t size) {
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void vector_fma(const float* a, const float* b, const float* c, float* result, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(result + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < size; ++i) {
        result[i] = a[i] * b[i] + c[i];
    }
}

// ============================================================================
// Aligned Variants (use _mm256_load_ps - requires 32-byte alignment)
// ============================================================================

float vector_sum_aligned(const float* data, size_t size) {
    // Debug-mode alignment check
    assert_avx2_aligned(data, "vector_sum_aligned");
    
    __m256 sum_vec = _mm256_setzero_ps();
    
    size_t i = 0;
    // Process 8 floats at a time using ALIGNED loads (faster!)
    for (; i + 8 <= size; i += 8) {
        __m256 v = _mm256_load_ps(data + i);  // ALIGNED load
        sum_vec = _mm256_add_ps(sum_vec, v);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum_vec, 1);
    __m128 lo = _mm256_castps256_ps128(sum_vec);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    // Handle remainder (scalar - can't use aligned for partial vectors)
    for (; i < size; ++i) {
        result += data[i];
    }
    
    return result;
}

float vector_dot_aligned(const float* a, const float* b, size_t size) {
    // Debug-mode alignment checks
    assert_avx2_aligned(a, "vector_dot_aligned (a)");
    assert_avx2_aligned(b, "vector_dot_aligned (b)");
    
    __m256 sum = _mm256_setzero_ps();
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_load_ps(a + i);  // ALIGNED load
        __m256 vb = _mm256_load_ps(b + i);  // ALIGNED load
        sum = _mm256_fmadd_ps(va, vb, sum);
    }
    
    // Horizontal sum
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(hi, lo);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);
    
    for (; i < size; ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

void threshold_crossing(
    const float* data,
    const float* thresholds,
    int32_t* crossings,
    size_t size
) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 vd = _mm256_loadu_ps(data + i);
        __m256 vt = _mm256_loadu_ps(thresholds + i);
        __m256 cmp = _mm256_cmp_ps(vd, vt, _CMP_LT_OQ);
        
        // Convert to int32 mask
        __m256i mask = _mm256_castps_si256(cmp);
        mask = _mm256_srli_epi32(mask, 31);  // Convert to 0 or 1
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(crossings + i), mask);
    }
    for (; i < size; ++i) {
        crossings[i] = (data[i] < thresholds[i]) ? 1 : 0;
    }
}

void compute_zscores(
    const float* data,
    const float* means,
    const float* stds,
    float* result,
    size_t size
) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 vd = _mm256_loadu_ps(data + i);
        __m256 vm = _mm256_loadu_ps(means + i);
        __m256 vs = _mm256_loadu_ps(stds + i);
        __m256 diff = _mm256_sub_ps(vd, vm);
        _mm256_storeu_ps(result + i, _mm256_div_ps(diff, vs));
    }
    for (; i < size; ++i) {
        result[i] = (data[i] - means[i]) / stds[i];
    }
}

void exponential_smooth(
    const float* current,
    const float* previous,
    float alpha,
    float* result,
    size_t size
) {
    __m256 a = _mm256_set1_ps(alpha);
    __m256 one_minus_a = _mm256_set1_ps(1.0f - alpha);
    
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 vc = _mm256_loadu_ps(current + i);
        __m256 vp = _mm256_loadu_ps(previous + i);
        __m256 r = _mm256_fmadd_ps(a, vc, _mm256_mul_ps(one_minus_a, vp));
        _mm256_storeu_ps(result + i, r);
    }
    for (; i < size; ++i) {
        result[i] = alpha * current[i] + (1.0f - alpha) * previous[i];
    }
}

void matrix_vector_mul(
    const float* matrix,
    const float* vector,
    float* result,
    size_t rows,
    size_t cols
) {
    for (size_t r = 0; r < rows; ++r) {
        result[r] = vector_dot(matrix + r * cols, vector, cols);
    }
}

#else  // Scalar fallback

float vector_sum(const float* data, size_t size) {
    return std::accumulate(data, data + size, 0.0f);
}

float vector_mean(const float* data, size_t size) {
    return vector_sum(data, size) / static_cast<float>(size);
}

float vector_variance(const float* data, size_t size) {
    float mean = vector_mean(data, size);
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float diff = data[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / static_cast<float>(size);
}

float vector_std(const float* data, size_t size) {
    return std::sqrt(vector_variance(data, size));
}

float vector_max(const float* data, size_t size) {
    return *std::max_element(data, data + size);
}

float vector_min(const float* data, size_t size) {
    return *std::min_element(data, data + size);
}

void vector_add(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) result[i] = a[i] + b[i];
}

void vector_sub(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) result[i] = a[i] - b[i];
}

void vector_mul(const float* a, const float* b, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) result[i] = a[i] * b[i];
}

void vector_scale(const float* a, float scalar, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) result[i] = a[i] * scalar;
}

float vector_dot(const float* a, const float* b, size_t size) {
    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) sum += a[i] * b[i];
    return sum;
}

void vector_fma(const float* a, const float* b, const float* c, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) result[i] = a[i] * b[i] + c[i];
}

void threshold_crossing(
    const float* data,
    const float* thresholds,
    int32_t* crossings,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        crossings[i] = (data[i] < thresholds[i]) ? 1 : 0;
    }
}

void compute_zscores(
    const float* data,
    const float* means,
    const float* stds,
    float* result,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = (data[i] - means[i]) / stds[i];
    }
}

void exponential_smooth(
    const float* current,
    const float* previous,
    float alpha,
    float* result,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = alpha * current[i] + (1.0f - alpha) * previous[i];
    }
}

void matrix_vector_mul(
    const float* matrix,
    const float* vector,
    float* result,
    size_t rows,
    size_t cols
) {
    for (size_t r = 0; r < rows; ++r) {
        result[r] = vector_dot(matrix + r * cols, vector, cols);
    }
}

#endif  // PHANTOMCORE_HAS_AVX2

// ============================================================================
// Channel Processor Implementation
// ============================================================================

float ChannelProcessor::compute_mean_rate(const AlignedSpikeData& spikes) {
    return vector_mean(spikes.data(), NUM_CHANNELS);
}

void ChannelProcessor::compute_zscores(
    const AlignedSpikeData& spikes,
    const AlignedSpikeData& means,
    const AlignedSpikeData& stds,
    AlignedSpikeData& result
) {
    simd::compute_zscores(
        spikes.data(), means.data(), stds.data(),
        result.data(), NUM_CHANNELS
    );
}

Vec2 ChannelProcessor::apply_decoder(
    const AlignedSpikeData& spikes,
    const std::array<float, NUM_CHANNELS>& weights_x,
    const std::array<float, NUM_CHANNELS>& weights_y,
    float bias_x,
    float bias_y
) {
    Vec2 result;
    result.x = vector_dot(spikes.data(), weights_x.data(), NUM_CHANNELS) + bias_x;
    result.y = vector_dot(spikes.data(), weights_y.data(), NUM_CHANNELS) + bias_y;
    return result;
}

void ChannelProcessor::update_statistics(
    const AlignedSpikeData& spikes,
    AlignedSpikeData& running_mean,
    AlignedSpikeData& running_var,
    size_t sample_count
) {
    // Welford's online algorithm for variance
    float n = static_cast<float>(sample_count + 1);
    
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        float delta = spikes[i] - running_mean[i];
        running_mean[i] += delta / n;
        float delta2 = spikes[i] - running_mean[i];
        running_var[i] += delta * delta2;
    }
}

}  // namespace simd
}  // namespace phantomcore
