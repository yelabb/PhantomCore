#include <gtest/gtest.h>
#include <phantomcore/spike_detector.hpp>
#include <random>

using namespace phantomcore;

class SpikeDetectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.threshold_std = -4.0f;
        config_.use_adaptive_threshold = true;
        channel_config_ = ChannelConfig::mc_maze();  // 142 channels
    }
    
    ChannelConfig channel_config_;
    SpikeDetectorConfig config_;
};

TEST_F(SpikeDetectorTest, CreateDefault) {
    SpikeDetector detector;
    EXPECT_EQ(detector.get_thresholds().size(), 142u);  // MC_Maze default
    EXPECT_EQ(detector.num_channels(), 142u);
}

TEST_F(SpikeDetectorTest, CreateWithChannelConfig) {
    // Test with Utah Array (96 channels)
    SpikeDetector detector(ChannelConfig::utah_array_96());
    EXPECT_EQ(detector.num_channels(), 96u);
    EXPECT_EQ(detector.get_thresholds().size(), 96u);
}

TEST_F(SpikeDetectorTest, CreateWithConfig) {
    SpikeDetector detector(channel_config_, config_);
    EXPECT_EQ(detector.config().threshold_std, -4.0f);
    EXPECT_EQ(detector.num_channels(), 142u);
}

TEST_F(SpikeDetectorTest, ProcessSpikeCounts) {
    SpikeDetector detector;
    
    SpikeCountArray counts{};
    counts[0] = 5;
    counts[1] = 3;
    counts[50] = 10;
    
    auto events = detector.process_spike_counts(counts, 0.0);
    
    // Should have detected spikes on channels with non-zero counts
    EXPECT_GE(events.size(), 3u);
}

TEST_F(SpikeDetectorTest, Reset) {
    SpikeDetector detector;
    
    SpikeCountArray counts{};
    counts[0] = 5;
    detector.process_spike_counts(counts, 0.0);
    
    auto stats_before = detector.get_stats();
    EXPECT_GT(stats_before.total_spikes_detected, 0u);
    
    detector.reset();
    
    auto stats_after = detector.get_stats();
    EXPECT_EQ(stats_after.total_spikes_detected, 0u);
}

TEST_F(SpikeDetectorTest, ChannelEnable) {
    SpikeDetector detector;
    
    detector.set_channel_enabled(0, false);
    
    SpikeCountArray counts{};
    counts[0] = 10;  // Disabled channel
    counts[1] = 10;  // Enabled channel
    
    auto events = detector.process_spike_counts(counts, 0.0);
    
    // Check that no events came from channel 0
    for (const auto& event : events) {
        EXPECT_NE(event.channel, 0u);
    }
}

TEST_F(SpikeDetectorTest, StatsTracking) {
    SpikeDetector detector;
    
    SpikeCountArray counts{};
    for (size_t i = 0; i < 142; ++i) {
        counts[i] = 1;
    }
    
    // Process multiple packets
    for (int i = 0; i < 100; ++i) {
        detector.process_spike_counts(counts, static_cast<double>(i) / 40.0);
    }
    
    auto stats = detector.get_stats();
    EXPECT_GT(stats.total_spikes_detected, 0u);
    EXPECT_GT(stats.mean_rate_hz, 0.0);
}

// Spike Sorter Tests

class SpikeSorterTest : public ::testing::Test {
protected:
    SpikeSorter::Config config_;
};

TEST_F(SpikeSorterTest, CreateDefault) {
    SpikeSorter sorter;
    EXPECT_FALSE(sorter.is_trained());
}

TEST_F(SpikeSorterTest, AddWaveform) {
    SpikeSorter sorter;
    
    Waveform wf;
    wf.samples.fill(0.0f);
    wf.channel = 0;
    wf.timestamp = 0.0;
    
    sorter.add_waveform(wf);
    // Should not throw
}

TEST_F(SpikeSorterTest, TrainRequiresData) {
    SpikeSorter sorter;
    
    // No data added, training should fail
    EXPECT_FALSE(sorter.train());
}

TEST_F(SpikeSorterTest, TrainWithData) {
    SpikeSorter::Config config;
    config.num_clusters = 2;
    SpikeSorter sorter(config);
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist1(0.0f, 1.0f);
    std::normal_distribution<float> dist2(5.0f, 1.0f);
    
    // Add waveforms from two clusters
    for (int i = 0; i < 100; ++i) {
        Waveform wf;
        if (i < 50) {
            for (auto& s : wf.samples) s = dist1(gen);
        } else {
            for (auto& s : wf.samples) s = dist2(gen);
        }
        wf.channel = 0;
        wf.timestamp = static_cast<double>(i);
        sorter.add_waveform(wf);
    }
    
    EXPECT_TRUE(sorter.train());
    EXPECT_TRUE(sorter.is_trained());
}

TEST_F(SpikeSorterTest, Classify) {
    SpikeSorter::Config config;
    config.num_clusters = 2;
    SpikeSorter sorter(config);
    
    // Add training data
    for (int i = 0; i < 100; ++i) {
        Waveform wf;
        float base = (i < 50) ? 0.0f : 10.0f;
        for (auto& s : wf.samples) s = base;
        wf.channel = 0;
        wf.timestamp = static_cast<double>(i);
        sorter.add_waveform(wf);
    }
    
    sorter.train();
    
    // Classify new waveforms
    Waveform test1, test2;
    test1.samples.fill(0.0f);
    test2.samples.fill(10.0f);
    
    uint32_t cluster1 = sorter.classify(test1);
    uint32_t cluster2 = sorter.classify(test2);
    
    // Should be assigned to different clusters
    EXPECT_NE(cluster1, cluster2);
}
