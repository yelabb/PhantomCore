/**
 * PhantomCore Latency Benchmark
 * 
 * Measures decode latency under various conditions to demonstrate
 * sub-millisecond performance. Generates detailed statistics.
 */

#include <phantomcore.hpp>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>
#include <algorithm>
#include <fstream>

using namespace phantomcore;

struct BenchmarkResult {
    std::string name;
    size_t iterations;
    LatencyStats stats;
};

void print_result(const BenchmarkResult& result) {
    std::cout << std::left << std::setw(30) << result.name
              << std::right
              << std::setw(10) << result.iterations
              << std::setw(12) << std::fixed << std::setprecision(2) << result.stats.mean_us
              << std::setw(12) << result.stats.std_us
              << std::setw(12) << result.stats.p50_us
              << std::setw(12) << result.stats.p95_us
              << std::setw(12) << result.stats.p99_us
              << std::setw(12) << result.stats.max_us
              << "\n";
}

void print_header() {
    std::cout << "\n";
    std::cout << std::left << std::setw(30) << "Benchmark"
              << std::right
              << std::setw(10) << "Iters"
              << std::setw(12) << "Mean(μs)"
              << std::setw(12) << "Std(μs)"
              << std::setw(12) << "P50(μs)"
              << std::setw(12) << "P95(μs)"
              << std::setw(12) << "P99(μs)"
              << std::setw(12) << "Max(μs)"
              << "\n";
    std::cout << std::string(100, '-') << "\n";
}

// Generate realistic spike count data
SpikeCountArray generate_spike_counts(std::mt19937& gen) {
    SpikeCountArray counts;
    std::poisson_distribution<int32_t> dist(3);  // Mean firing rate
    
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        counts[i] = dist(gen);
    }
    
    return counts;
}

// Benchmark: Kalman Decoder
BenchmarkResult benchmark_kalman_decoder(size_t iterations) {
    std::mt19937 gen(42);
    KalmanDecoder decoder;
    LatencyTracker tracker;
    
    // Warm-up
    for (size_t i = 0; i < 100; ++i) {
        auto counts = generate_spike_counts(gen);
        decoder.decode(counts);
    }
    decoder.reset();
    
    // Benchmark
    for (size_t i = 0; i < iterations; ++i) {
        auto counts = generate_spike_counts(gen);
        
        auto start = Clock::now();
        decoder.decode(counts);
        tracker.record(Clock::now() - start);
    }
    
    return {"Kalman Decoder", iterations, tracker.get_stats()};
}

// Benchmark: Linear Decoder
BenchmarkResult benchmark_linear_decoder(size_t iterations) {
    std::mt19937 gen(42);
    LinearDecoder decoder;
    LatencyTracker tracker;
    
    // Warm-up
    for (size_t i = 0; i < 100; ++i) {
        auto counts = generate_spike_counts(gen);
        decoder.decode(counts);
    }
    decoder.reset();
    
    // Benchmark
    for (size_t i = 0; i < iterations; ++i) {
        auto counts = generate_spike_counts(gen);
        
        auto start = Clock::now();
        decoder.decode(counts);
        tracker.record(Clock::now() - start);
    }
    
    return {"Linear Decoder", iterations, tracker.get_stats()};
}

// Benchmark: Spike Detector
BenchmarkResult benchmark_spike_detector(size_t iterations) {
    std::mt19937 gen(42);
    SpikeDetector detector;
    LatencyTracker tracker;
    
    // Generate raw samples (simulating continuous recording)
    std::normal_distribution<float> noise_dist(0.0f, 1.0f);
    
    // Warm-up
    for (size_t i = 0; i < 100; ++i) {
        auto counts = generate_spike_counts(gen);
        detector.process_spike_counts(counts, static_cast<double>(i) / 40.0);
    }
    detector.reset();
    
    // Benchmark
    for (size_t i = 0; i < iterations; ++i) {
        auto counts = generate_spike_counts(gen);
        
        auto start = Clock::now();
        detector.process_spike_counts(counts, static_cast<double>(i) / 40.0);
        tracker.record(Clock::now() - start);
    }
    
    return {"Spike Detector", iterations, tracker.get_stats()};
}

// Benchmark: SIMD Vector Operations
BenchmarkResult benchmark_simd_dot_product(size_t iterations) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    AlignedSpikeData a, b;
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        a[i] = dist(gen);
        b[i] = dist(gen);
    }
    
    LatencyTracker tracker;
    volatile float result;  // Prevent optimization
    
    for (size_t i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        result = simd::vector_dot(a.data(), b.data(), NUM_CHANNELS);
        tracker.record(Clock::now() - start);
    }
    
    (void)result;
    return {"SIMD Dot Product (142-dim)", iterations, tracker.get_stats()};
}

// Benchmark: SIMD Z-Score Computation
BenchmarkResult benchmark_simd_zscore(size_t iterations) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    AlignedSpikeData data, means, stds, result;
    for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        data[i] = dist(gen) * 10.0f;
        means[i] = dist(gen);
        stds[i] = 1.0f + std::abs(dist(gen));
    }
    
    LatencyTracker tracker;
    
    for (size_t i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        simd::ChannelProcessor::compute_zscores(data, means, stds, result);
        tracker.record(Clock::now() - start);
    }
    
    return {"SIMD Z-Score (142 ch)", iterations, tracker.get_stats()};
}

// Benchmark: Ring Buffer Operations
BenchmarkResult benchmark_ring_buffer(size_t iterations) {
    RingBuffer<NeuralPacket, 64> buffer;
    NeuralPacket packet;
    LatencyTracker tracker;
    
    for (size_t i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        buffer.push(packet);
        auto opt = buffer.pop();
        tracker.record(Clock::now() - start);
        (void)opt;
    }
    
    return {"Ring Buffer Push+Pop", iterations, tracker.get_stats()};
}

// Benchmark: Full Pipeline
BenchmarkResult benchmark_full_pipeline(size_t iterations) {
    std::mt19937 gen(42);
    
    SpikeDetector detector;
    KalmanDecoder decoder;
    LatencyTracker tracker;
    
    // Warm-up
    for (size_t i = 0; i < 100; ++i) {
        auto counts = generate_spike_counts(gen);
        detector.process_spike_counts(counts, static_cast<double>(i) / 40.0);
        decoder.decode(counts);
    }
    detector.reset();
    decoder.reset();
    
    // Benchmark
    for (size_t i = 0; i < iterations; ++i) {
        auto counts = generate_spike_counts(gen);
        
        auto start = Clock::now();
        
        // Full pipeline: detect + decode
        detector.process_spike_counts(counts, static_cast<double>(i) / 40.0);
        decoder.decode(counts);
        
        tracker.record(Clock::now() - start);
    }
    
    return {"Full Pipeline (Detect+Decode)", iterations, tracker.get_stats()};
}

int main() {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PhantomCore Latency Benchmark Suite                        ║
║                    Sub-Millisecond Neural Processing                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    phantomcore::initialize();
    std::cout << "Build: " << phantomcore::build_info() << "\n";
    std::cout << "SIMD:  " << simd::simd_info() << "\n";
    
    const size_t ITERATIONS = 10000;
    
    std::vector<BenchmarkResult> results;
    
    std::cout << "\nRunning benchmarks with " << ITERATIONS << " iterations each...\n";
    
    // Run benchmarks
    std::cout << "  [1/7] SIMD Dot Product..." << std::flush;
    results.push_back(benchmark_simd_dot_product(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [2/7] SIMD Z-Score..." << std::flush;
    results.push_back(benchmark_simd_zscore(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [3/7] Ring Buffer..." << std::flush;
    results.push_back(benchmark_ring_buffer(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [4/7] Spike Detector..." << std::flush;
    results.push_back(benchmark_spike_detector(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [5/7] Linear Decoder..." << std::flush;
    results.push_back(benchmark_linear_decoder(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [6/7] Kalman Decoder..." << std::flush;
    results.push_back(benchmark_kalman_decoder(ITERATIONS));
    std::cout << " done\n";
    
    std::cout << "  [7/7] Full Pipeline..." << std::flush;
    results.push_back(benchmark_full_pipeline(ITERATIONS));
    std::cout << " done\n";
    
    // Print results
    print_header();
    for (const auto& result : results) {
        print_result(result);
    }
    std::cout << std::string(100, '-') << "\n";
    
    // Summary
    std::cout << "\n=== Summary ===\n\n";
    
    double full_pipeline_mean = results.back().stats.mean_us;
    double full_pipeline_p99 = results.back().stats.p99_us;
    
    std::cout << "Full Pipeline Performance:\n";
    std::cout << "  Mean latency:     " << std::fixed << std::setprecision(2) 
              << full_pipeline_mean << " μs";
    if (full_pipeline_mean < 1000.0) {
        std::cout << " ✓ SUB-MILLISECOND";
    }
    std::cout << "\n";
    
    std::cout << "  P99 latency:      " << full_pipeline_p99 << " μs";
    if (full_pipeline_p99 < 1000.0) {
        std::cout << " ✓ SUB-MILLISECOND";
    }
    std::cout << "\n\n";
    
    // Calculate theoretical throughput
    double packets_per_second = 1'000'000.0 / full_pipeline_mean;
    std::cout << "Theoretical Throughput:\n";
    std::cout << "  " << std::fixed << std::setprecision(0) 
              << packets_per_second << " packets/sec\n";
    std::cout << "  " << packets_per_second / 40.0 << "x real-time (at 40Hz)\n\n";
    
    // Real-time capability
    double margin = (25000.0 - full_pipeline_mean) / 25000.0 * 100.0;  // 25ms = 40Hz period
    std::cout << "Real-Time Margin:\n";
    std::cout << "  " << std::fixed << std::setprecision(1) << margin 
              << "% headroom for 40Hz streaming\n";
    
    phantomcore::shutdown();
    
    return 0;
}
