#include <benchmark/benchmark.h>
#include <phantomcore/kalman_decoder.hpp>
#include <random>

using namespace phantomcore;

static void BM_KalmanDecoder_Decode(benchmark::State& state) {
    KalmanDecoder decoder;
    
    std::mt19937 gen(42);
    std::poisson_distribution<int32_t> dist(3);
    
    SpikeCountArray counts;
    for (auto& c : counts) c = dist(gen);
    
    for (auto _ : state) {
        auto output = decoder.decode(counts);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_KalmanDecoder_Decode);

static void BM_KalmanDecoder_DecodeAligned(benchmark::State& state) {
    KalmanDecoder decoder;
    
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.0f, 10.0f);
    
    AlignedSpikeData counts;
    for (auto& c : counts.counts) c = dist(gen);
    
    for (auto _ : state) {
        auto output = decoder.decode(counts);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_KalmanDecoder_DecodeAligned);

static void BM_KalmanDecoder_Predict(benchmark::State& state) {
    KalmanDecoder decoder;
    
    for (auto _ : state) {
        auto output = decoder.predict();
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_KalmanDecoder_Predict);

static void BM_LinearDecoder_Decode(benchmark::State& state) {
    LinearDecoder::Config config;
    config.weights_x.fill(0.01f);
    config.weights_y.fill(0.02f);
    
    LinearDecoder decoder(config);
    
    std::mt19937 gen(42);
    std::poisson_distribution<int32_t> dist(3);
    
    SpikeCountArray counts;
    for (auto& c : counts) c = dist(gen);
    
    for (auto _ : state) {
        auto output = decoder.decode(counts);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_LinearDecoder_Decode);

BENCHMARK_MAIN();
