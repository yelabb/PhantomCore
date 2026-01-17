#include <benchmark/benchmark.h>
#include <phantomcore/simd_utils.hpp>
#include <random>

using namespace phantomcore;
using namespace phantomcore::simd;

static void BM_VectorSum_142(benchmark::State& state) {
    AlignedSpikeData data;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& d : data.counts) d = dist(gen);
    
    for (auto _ : state) {
        float result = vector_sum(data.data(), NUM_CHANNELS);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorSum_142);

static void BM_VectorDot_142(benchmark::State& state) {
    AlignedSpikeData a, b;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }
    
    for (auto _ : state) {
        float result = vector_dot(a.data(), b.data(), NUM_CHANNELS);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorDot_142);

static void BM_VectorMean_142(benchmark::State& state) {
    AlignedSpikeData data;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& d : data.counts) d = dist(gen);
    
    for (auto _ : state) {
        float result = vector_mean(data.data(), NUM_CHANNELS);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorMean_142);

static void BM_VectorVariance_142(benchmark::State& state) {
    AlignedSpikeData data;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (auto& d : data.counts) d = dist(gen);
    
    for (auto _ : state) {
        float result = vector_variance(data.data(), NUM_CHANNELS);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_VectorVariance_142);

static void BM_ZScore_142(benchmark::State& state) {
    AlignedSpikeData data, means, stds, result;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        data[i] = dist(gen);
        means[i] = dist(gen);
        stds[i] = 1.0f + std::abs(dist(gen));
    }
    
    for (auto _ : state) {
        ChannelProcessor::compute_zscores(data, means, stds, result);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ZScore_142);

static void BM_ApplyDecoder_142(benchmark::State& state) {
    AlignedSpikeData spikes;
    std::array<float, NUM_CHANNELS> weights_x, weights_y;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        spikes[i] = std::abs(dist(gen)) * 10.0f;
        weights_x[i] = dist(gen) * 0.01f;
        weights_y[i] = dist(gen) * 0.01f;
    }
    
    for (auto _ : state) {
        Vec2 result = ChannelProcessor::apply_decoder(
            spikes, weights_x, weights_y, 0.0f, 0.0f
        );
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_ApplyDecoder_142);

static void BM_ThresholdCrossing_142(benchmark::State& state) {
    std::array<float, NUM_CHANNELS> data;
    std::array<float, NUM_CHANNELS> thresholds;
    std::array<int32_t, NUM_CHANNELS> crossings;
    
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        data[i] = dist(gen);
        thresholds[i] = -3.0f;
    }
    
    for (auto _ : state) {
        threshold_crossing(data.data(), thresholds.data(), crossings.data(), NUM_CHANNELS);
        benchmark::DoNotOptimize(crossings);
    }
}
BENCHMARK(BM_ThresholdCrossing_142);

// Compare SIMD vs scalar for various sizes
static void BM_VectorDot_Size(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<float> a(size), b(size);
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < size; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }
    
    for (auto _ : state) {
        float result = vector_dot(a.data(), b.data(), size);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 2);
}
BENCHMARK(BM_VectorDot_Size)->Range(8, 8192);

// Note: BENCHMARK_MAIN() is defined in bench_spike_detector.cpp
