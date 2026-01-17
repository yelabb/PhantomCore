#include <benchmark/benchmark.h>
#include <phantomcore/spike_detector.hpp>
#include <random>

using namespace phantomcore;

static void BM_SpikeDetector_ProcessSpikeCounts(benchmark::State& state) {
    SpikeDetector detector;
    
    std::mt19937 gen(42);
    std::poisson_distribution<int32_t> dist(3);
    
    SpikeCountArray counts;
    for (auto& c : counts) c = dist(gen);
    
    double timestamp = 0.0;
    for (auto _ : state) {
        auto events = detector.process_spike_counts(counts, timestamp);
        benchmark::DoNotOptimize(events);
        timestamp += 0.025;
    }
}
BENCHMARK(BM_SpikeDetector_ProcessSpikeCounts);

static void BM_SpikeDetector_ProcessBatch(benchmark::State& state) {
    SpikeDetector detector;
    
    const size_t batch_size = state.range(0);
    const size_t num_channels = NUM_CHANNELS;
    
    std::vector<float> samples(batch_size * num_channels);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (auto& s : samples) s = dist(gen);
    
    for (auto _ : state) {
        auto events = detector.process_batch(
            samples, batch_size, num_channels, 0.0, 30000.0
        );
        benchmark::DoNotOptimize(events);
    }
    
    state.SetItemsProcessed(state.iterations() * batch_size);
}
BENCHMARK(BM_SpikeDetector_ProcessBatch)->Range(32, 1024);

BENCHMARK_MAIN();
