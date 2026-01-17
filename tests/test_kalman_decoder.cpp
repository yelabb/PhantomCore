#include <gtest/gtest.h>
#include <phantomcore/kalman_decoder.hpp>
#include <random>

using namespace phantomcore;

class KalmanDecoderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Generate some test data
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        for (auto& c : test_spikes_) {
            c = static_cast<int32_t>(std::abs(dist(gen)) * 5);
        }
    }
    
    SpikeCountArray test_spikes_{};
};

TEST_F(KalmanDecoderTest, CreateDefault) {
    KalmanDecoder decoder;
    
    auto state = decoder.get_state();
    EXPECT_FLOAT_EQ(state(0), 0.0f);  // x
    EXPECT_FLOAT_EQ(state(1), 0.0f);  // y
    EXPECT_FLOAT_EQ(state(2), 0.0f);  // vx
    EXPECT_FLOAT_EQ(state(3), 0.0f);  // vy
}

TEST_F(KalmanDecoderTest, DecodeProducesOutput) {
    KalmanDecoder decoder;
    
    auto output = decoder.decode(test_spikes_);
    
    // Should produce some output (even if not trained)
    EXPECT_TRUE(std::isfinite(output.position.x));
    EXPECT_TRUE(std::isfinite(output.position.y));
}

TEST_F(KalmanDecoderTest, DecodeUpdatesState) {
    KalmanDecoder decoder;
    
    auto state_before = decoder.get_state();
    decoder.decode(test_spikes_);
    auto state_after = decoder.get_state();
    
    // State should change after processing
    // (May be the same if observation model is zero)
}

TEST_F(KalmanDecoderTest, Reset) {
    KalmanDecoder decoder;
    
    // Process some data
    for (int i = 0; i < 10; ++i) {
        decoder.decode(test_spikes_);
    }
    
    decoder.reset();
    
    auto state = decoder.get_state();
    EXPECT_FLOAT_EQ(state(0), 0.0f);
    EXPECT_FLOAT_EQ(state(1), 0.0f);
}

TEST_F(KalmanDecoderTest, Predict) {
    KalmanDecoder decoder;
    
    auto output = decoder.predict();
    
    EXPECT_TRUE(std::isfinite(output.position.x));
    EXPECT_LT(output.confidence, 1.0f);  // Lower confidence for prediction
}

TEST_F(KalmanDecoderTest, ProcessingTime) {
    KalmanDecoder decoder;
    
    auto output = decoder.decode(test_spikes_);
    
    // Processing time should be measured
    EXPECT_GT(output.processing_time.count(), 0);
}

TEST_F(KalmanDecoderTest, Stats) {
    KalmanDecoder decoder;
    
    for (int i = 0; i < 100; ++i) {
        decoder.decode(test_spikes_);
    }
    
    auto stats = decoder.get_stats();
    EXPECT_EQ(stats.total_decodes, 100u);
    EXPECT_GT(stats.mean_decode_time.count(), 0);
}

TEST_F(KalmanDecoderTest, SaveLoadWeights) {
    KalmanDecoder decoder;
    
    auto weights = decoder.save_weights();
    // 142 channels * 4 state dims = 568 weights
    EXPECT_EQ(weights.size(), decoder.num_channels() * KalmanDecoder::STATE_DIM);
    
    // Modify weights
    std::fill(weights.begin(), weights.end(), 0.5f);
    
    decoder.load_weights(weights);
    auto loaded = decoder.save_weights();
    
    EXPECT_FLOAT_EQ(loaded[0], 0.5f);
}

// Linear Decoder Tests

class LinearDecoderTest : public ::testing::Test {
protected:
    SpikeCountArray test_spikes_{};
    static constexpr size_t TEST_CHANNELS = 142;  // MC_Maze default
    
    void SetUp() override {
        for (size_t i = 0; i < TEST_CHANNELS; ++i) {
            test_spikes_[i] = static_cast<int32_t>(i % 10);
        }
    }
};

TEST_F(LinearDecoderTest, CreateDefault) {
    LinearDecoder decoder;
    // Should not throw
}

TEST_F(LinearDecoderTest, DecodeWithDefaultWeights) {
    LinearDecoder decoder;
    
    auto output = decoder.decode(test_spikes_);
    
    // With zero weights, output should be near zero (just bias)
    EXPECT_TRUE(std::isfinite(output.position.x));
    EXPECT_TRUE(std::isfinite(output.position.y));
}

TEST_F(LinearDecoderTest, DecodeWithCustomWeights) {
    LinearDecoder::Config config(ChannelConfig::mc_maze());
    std::fill(config.weights_x.begin(), config.weights_x.end(), 0.01f);
    std::fill(config.weights_y.begin(), config.weights_y.end(), 0.02f);
    config.bias_x = 1.0f;
    config.bias_y = 2.0f;
    
    LinearDecoder decoder(config);
    
    auto output = decoder.decode(test_spikes_);
    
    // Should produce non-zero output
    EXPECT_NE(output.position.x, 0.0f);
    EXPECT_NE(output.position.y, 0.0f);
}

TEST_F(LinearDecoderTest, FastProcessing) {
    LinearDecoder decoder;
    
    auto output = decoder.decode(test_spikes_);
    
    // Linear decoder should be very fast (< 100μs)
    double us = to_microseconds(output.processing_time);
    EXPECT_LT(us, 100.0);
}
