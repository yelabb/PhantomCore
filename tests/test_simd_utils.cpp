#include <gtest/gtest.h>
#include <phantomcore/simd_utils.hpp>
#include <random>
#include <cmath>

using namespace phantomcore;
using namespace phantomcore::simd;

class SIMDTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
        
        for (size_t i = 0; i < test_size_; ++i) {
            data_a_[i] = dist(gen);
            data_b_[i] = dist(gen);
        }
    }
    
    static constexpr size_t test_size_ = 256;
    float data_a_[test_size_];
    float data_b_[test_size_];
    float result_[test_size_];
};

TEST_F(SIMDTest, HasSimdInfo) {
    const char* info = simd_info();
    EXPECT_NE(info, nullptr);
    EXPECT_GT(strlen(info), 0u);
}

TEST_F(SIMDTest, VectorSum) {
    float expected = 0.0f;
    for (size_t i = 0; i < test_size_; ++i) {
        expected += data_a_[i];
    }
    
    float result = vector_sum(data_a_, test_size_);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-5f);
}

TEST_F(SIMDTest, VectorMean) {
    float expected = 0.0f;
    for (size_t i = 0; i < test_size_; ++i) {
        expected += data_a_[i];
    }
    expected /= static_cast<float>(test_size_);
    
    float result = vector_mean(data_a_, test_size_);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-5f + 1e-6f);
}

TEST_F(SIMDTest, VectorVariance) {
    float mean = vector_mean(data_a_, test_size_);
    
    float expected = 0.0f;
    for (size_t i = 0; i < test_size_; ++i) {
        float diff = data_a_[i] - mean;
        expected += diff * diff;
    }
    expected /= static_cast<float>(test_size_);
    
    float result = vector_variance(data_a_, test_size_);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-4f + 1e-5f);
}

TEST_F(SIMDTest, VectorMax) {
    float expected = data_a_[0];
    for (size_t i = 1; i < test_size_; ++i) {
        expected = std::max(expected, data_a_[i]);
    }
    
    float result = vector_max(data_a_, test_size_);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDTest, VectorMin) {
    float expected = data_a_[0];
    for (size_t i = 1; i < test_size_; ++i) {
        expected = std::min(expected, data_a_[i]);
    }
    
    float result = vector_min(data_a_, test_size_);
    EXPECT_FLOAT_EQ(result, expected);
}

TEST_F(SIMDTest, VectorAdd) {
    vector_add(data_a_, data_b_, result_, test_size_);
    
    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(result_[i], data_a_[i] + data_b_[i]);
    }
}

TEST_F(SIMDTest, VectorSub) {
    vector_sub(data_a_, data_b_, result_, test_size_);
    
    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(result_[i], data_a_[i] - data_b_[i]);
    }
}

TEST_F(SIMDTest, VectorMul) {
    vector_mul(data_a_, data_b_, result_, test_size_);
    
    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(result_[i], data_a_[i] * data_b_[i]);
    }
}

TEST_F(SIMDTest, VectorScale) {
    const float scalar = 2.5f;
    vector_scale(data_a_, scalar, result_, test_size_);
    
    for (size_t i = 0; i < test_size_; ++i) {
        EXPECT_FLOAT_EQ(result_[i], data_a_[i] * scalar);
    }
}

TEST_F(SIMDTest, VectorDot) {
    float expected = 0.0f;
    for (size_t i = 0; i < test_size_; ++i) {
        expected += data_a_[i] * data_b_[i];
    }
    
    float result = vector_dot(data_a_, data_b_, test_size_);
    EXPECT_NEAR(result, expected, std::abs(expected) * 1e-5f + 1e-4f);
}

TEST_F(SIMDTest, ThresholdCrossing) {
    float data[16] = {-5, -3, -1, 0, 1, 3, 5, 7, -2, -4, 2, 4, -6, 6, 0, -1};
    float thresholds[16];
    int32_t crossings[16];
    
    for (int i = 0; i < 16; ++i) {
        thresholds[i] = 0.0f;  // All thresholds at 0
    }
    
    threshold_crossing(data, thresholds, crossings, 16);
    
    for (int i = 0; i < 16; ++i) {
        int expected = (data[i] < thresholds[i]) ? 1 : 0;
        EXPECT_EQ(crossings[i], expected);
    }
}

TEST_F(SIMDTest, ComputeZScores) {
    float data[8] = {2, 4, 6, 8, 10, 12, 14, 16};
    float means[8] = {5, 5, 5, 5, 5, 5, 5, 5};
    float stds[8] = {2, 2, 2, 2, 2, 2, 2, 2};
    float result[8];
    
    compute_zscores(data, means, stds, result, 8);
    
    for (int i = 0; i < 8; ++i) {
        float expected = (data[i] - means[i]) / stds[i];
        EXPECT_FLOAT_EQ(result[i], expected);
    }
}

TEST_F(SIMDTest, ExponentialSmooth) {
    float current[8] = {10, 20, 30, 40, 50, 60, 70, 80};
    float previous[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float result[8];
    float alpha = 0.5f;
    
    exponential_smooth(current, previous, alpha, result, 8);
    
    for (int i = 0; i < 8; ++i) {
        float expected = alpha * current[i] + (1.0f - alpha) * previous[i];
        EXPECT_FLOAT_EQ(result[i], expected);
    }
}

// Channel Processor Tests

constexpr size_t TEST_CHANNELS = 142;  // MC_Maze default for legacy tests

TEST(ChannelProcessorTest, ComputeMeanRate) {
    AlignedSpikeData spikes;
    for (size_t i = 0; i < TEST_CHANNELS; ++i) {
        spikes[i] = 2.0f;
    }
    
    float mean = ChannelProcessor::compute_mean_rate(spikes);
    EXPECT_FLOAT_EQ(mean, 2.0f);
}

TEST(ChannelProcessorTest, ComputeMeanRateDynamic) {
    SpikeData spikes(96);  // Utah Array
    for (size_t i = 0; i < 96; ++i) {
        spikes[i] = 3.0f;
    }
    
    float mean = ChannelProcessor::compute_mean_rate(spikes);
    EXPECT_FLOAT_EQ(mean, 3.0f);
}

TEST(ChannelProcessorTest, ApplyDecoder) {
    AlignedSpikeData spikes;
    spikes.counts.fill(1.0f);
    
    std::array<float, TEST_CHANNELS> weights_x, weights_y;
    weights_x.fill(0.01f);
    weights_y.fill(0.02f);
    
    float bias_x = 1.0f;
    float bias_y = 2.0f;
    
    Vec2 result = ChannelProcessor::apply_decoder(
        spikes, weights_x, weights_y, bias_x, bias_y
    );
    
    // Expected: sum(spikes * weights) + bias
    float expected_x = TEST_CHANNELS * 1.0f * 0.01f + 1.0f;
    float expected_y = TEST_CHANNELS * 1.0f * 0.02f + 2.0f;
    
    EXPECT_NEAR(result.x, expected_x, 1e-4f);
    EXPECT_NEAR(result.y, expected_y, 1e-4f);
}

TEST(ChannelProcessorTest, ApplyDecoderDynamic) {
    SpikeData spikes(96);
    std::vector<float> weights_x(96, 0.01f);
    std::vector<float> weights_y(96, 0.02f);
    
    for (size_t i = 0; i < 96; ++i) {
        spikes[i] = 1.0f;
    }
    
    Vec2 result = ChannelProcessor::apply_decoder(
        spikes.span(), weights_x, weights_y, 1.0f, 2.0f
    );
    
    float expected_x = 96 * 1.0f * 0.01f + 1.0f;
    float expected_y = 96 * 1.0f * 0.02f + 2.0f;
    
    EXPECT_NEAR(result.x, expected_x, 1e-4f);
    EXPECT_NEAR(result.y, expected_y, 1e-4f);
}
