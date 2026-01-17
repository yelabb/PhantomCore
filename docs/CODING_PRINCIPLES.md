# PhantomCore Coding Principles

> **Mission**: Deliver sub-millisecond neural decoding with zero compromises on reliability.

This document defines the engineering principles that govern all PhantomCore development. Every contributor must internalize these before writing code.

---

## 🎯 Core Tenets

### 1. Latency is Sacred

```
Every microsecond matters. Every allocation is suspect. Every branch is a risk.
```

- **Target**: < 15μs full pipeline (spike detection + decode)
- **Hard limit**: No operation may exceed 100μs in the hot path
- **Measure first**: Profile before and after every change
- **No guessing**: Use `LatencyTracker` for all timing claims

### 2. Correctness Before Speed

```
A fast wrong answer kills patients. A slow correct answer saves them.
```

- Validate all inputs at API boundaries
- Use `assert()` liberally in debug builds
- Write tests BEFORE optimization
- Never sacrifice numerical stability for speed

### 3. Predictable Over Fast

```
P99 latency matters more than mean latency.
```

- Avoid dynamic allocation in hot paths
- No exceptions in real-time code (use `std::expected` or error codes)
- Pre-allocate all buffers during initialization
- Lock-free structures over mutexes

---

## 🏗️ Architecture Principles

### Separation of Concerns

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Data Layer    │ ──▶ │ Processing Layer│ ──▶ │   Output Layer  │
│  (StreamClient) │     │ (Decoders/SIMD) │     │ (Callbacks/API) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

- **Data Layer**: Handles I/O, parsing, buffering
- **Processing Layer**: Pure computation, no I/O, no state mutation outside owned data
- **Output Layer**: Delivers results, handles user callbacks

### Component Independence

Each component must be:
- **Testable in isolation** - No hidden dependencies
- **Configurable at runtime** - Use `Config` structs, not compile-time constants
- **Replaceable** - Interfaces over implementations

### Memory Hierarchy Awareness

```cpp
// ❌ BAD: Cache-hostile access pattern
for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
    for (size_t t = 0; t < NUM_SAMPLES; ++t) {
        process(data[t][ch]);  // Stride = NUM_CHANNELS * sizeof(float)
    }
}

// ✅ GOOD: Cache-friendly access
for (size_t t = 0; t < NUM_SAMPLES; ++t) {
    for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        process(data[t][ch]);  // Sequential access
    }
}
```

---

## 💻 Code Style

### Naming Conventions

| Element | Style | Example |
|---------|-------|---------|
| Classes | PascalCase | `KalmanDecoder` |
| Functions | snake_case | `process_sample()` |
| Member variables | snake_case_ | `sample_rate_` |
| Constants | SCREAMING_SNAKE | `MAX_CHANNELS` |
| Template params | PascalCase | `typename BufferType` |
| Namespaces | lowercase | `phantomcore::simd` |

### File Organization

```
include/phantomcore/
    component.hpp      # Public API only
src/
    component.cpp      # Implementation
    component_impl.hpp # Private implementation details (if needed)
tests/
    test_component.cpp
benchmarks/
    bench_component.cpp
```

### Header Structure

```cpp
#pragma once

// Standard library (alphabetical)
#include <memory>
#include <span>
#include <vector>

// Third-party (alphabetical)
#include <Eigen/Dense>

// Project headers (alphabetical)
#include "phantomcore/types.hpp"

namespace phantomcore {

// Forward declarations first
class OtherComponent;

// Constants
constexpr size_t BUFFER_SIZE = 1024;

// Main class/function declarations
class Component {
public:
    // Types
    // Constructors/Destructor
    // Public methods
    // Operators
    
private:
    // Private methods
    // Member variables (trailing underscore)
};

}  // namespace phantomcore
```

---

## ⚡ Performance Rules

### Rule 1: No Allocations in Hot Paths

```cpp
// ❌ BAD: Allocates on every call
std::vector<float> process(std::span<const float> input) {
    std::vector<float> result(input.size());  // ALLOCATION!
    // ...
    return result;
}

// ✅ GOOD: Pre-allocated output buffer
void process(std::span<const float> input, std::span<float> output) {
    assert(output.size() >= input.size());
    // ...
}
```

### Rule 2: SIMD-Friendly Data Layout

```cpp
// ❌ BAD: Array of Structs (AoS)
struct Spike { float time; float amplitude; int channel; };
std::vector<Spike> spikes;

// ✅ GOOD: Struct of Arrays (SoA) for SIMD
struct Spikes {
    std::vector<float> times;
    std::vector<float> amplitudes;
    std::vector<int> channels;
};
```

### Rule 3: Alignment Matters

```cpp
// All SIMD buffers must be 32-byte aligned for AVX2
alignas(32) float buffer[256];

// Or use our aligned allocator
AlignedVector<float> buffer(256);
```

### Rule 4: Branch Prediction

```cpp
// ❌ BAD: Unpredictable branch in tight loop
for (size_t i = 0; i < n; ++i) {
    if (data[i] > threshold) {  // Unpredictable!
        count++;
    }
}

// ✅ GOOD: Branchless where possible
for (size_t i = 0; i < n; ++i) {
    count += (data[i] > threshold);  // No branch
}
```

### Rule 5: Measure Everything

```cpp
// Use LatencyTracker for all performance-critical paths
LatencyTracker tracker;

auto start = Clock::now();
auto result = decoder.decode(spikes);
tracker.record(Clock::now() - start);

// Assert timing requirements
auto stats = tracker.get_stats();
assert(stats.p99_us < 50.0 && "P99 latency exceeded budget!");
```

---

## 🔒 Safety Requirements

### Thread Safety

- **Shared data**: Use `std::atomic` or lock-free structures
- **Callbacks**: Document thread context clearly
- **State mutation**: Single-writer principle

```cpp
/// @thread_safety This callback is invoked on the WebSocket thread.
///                Handler must be fast (< 100μs) and non-blocking.
void on_packet(PacketCallback callback);
```

### Error Handling

```cpp
// ❌ BAD: Exceptions in real-time code
float decode(const SpikeData& data) {
    if (data.empty()) throw std::invalid_argument("Empty data");
    // ...
}

// ✅ GOOD: Error codes or std::expected
std::expected<float, DecodeError> decode(const SpikeData& data) {
    if (data.empty()) return std::unexpected(DecodeError::EmptyInput);
    // ...
}
```

### Resource Management

- **RAII everywhere**: No manual `new`/`delete`
- **Unique ownership**: Prefer `std::unique_ptr`
- **Non-copyable by default**: Delete copy constructor unless needed

```cpp
class Decoder {
public:
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;
    Decoder(Decoder&&) noexcept = default;
    Decoder& operator=(Decoder&&) noexcept = default;
};
```

---

## 🧪 Testing Requirements

### Unit Tests

- Every public method must have tests
- Test edge cases: empty input, max channels, NaN/Inf values
- Use Google Test framework

```cpp
TEST(KalmanDecoder, HandlesEmptyInput) {
    KalmanDecoder decoder;
    SpikeData empty(0);
    auto result = decoder.decode(empty);
    EXPECT_FALSE(result.has_value());
}
```

### Performance Tests

- Benchmark every algorithm
- Track P50, P95, P99 latencies
- Fail CI if P99 exceeds threshold

```cpp
static void BM_Decode(benchmark::State& state) {
    KalmanDecoder decoder;
    SpikeData data(142);
    
    for (auto _ : state) {
        auto result = decoder.decode(data);
        benchmark::DoNotOptimize(result);
    }
}
BENCHMARK(BM_Decode)->MinTime(1.0);
```

### Integration Tests

- Test full pipeline: StreamClient → Decoder → Output
- Simulate dropped packets, reconnections
- Validate numerical accuracy against reference implementation

---

## 📋 Enhancement Implementation Checklist

When implementing any of the 3 planned enhancements, follow this checklist:

### Pre-Implementation

- [ ] Write design doc with latency budget
- [ ] Define public API (header file first)
- [ ] Identify hot paths and allocation points
- [ ] Plan SIMD/GPU strategy

### Implementation

- [ ] Implement with full error handling
- [ ] Add `LatencyTracker` instrumentation
- [ ] Write unit tests (>90% coverage)
- [ ] Write benchmarks

### Post-Implementation

- [ ] Profile on target hardware
- [ ] Verify P99 < latency budget
- [ ] Update documentation
- [ ] Add example usage

---

## 🚀 Enhancement-Specific Guidelines

### GPU/CUDA Acceleration

```cpp
// GPU operations must have CPU fallback
#ifdef PHANTOMCORE_ENABLE_CUDA
    cuda_decode(d_spikes, d_output, stream);
#else
    cpu_decode(spikes, output);
#endif

// Async operations require explicit synchronization points
cudaStreamSynchronize(stream);  // Document where this happens!
```

### Adaptive Calibration

```cpp
// Online updates must be atomic and bounded
void update_weights(const Observation& obs) {
    // Budget: < 5μs per update
    // Use rank-1 update, not full matrix solve
    weights_ += learning_rate_ * outer_product(error, obs);
}
```

### Neural Network Decoder

```cpp
// ONNX Runtime sessions are expensive - create once
class NeuralNetDecoder {
    Ort::Session session_;  // Created in constructor
    
    // Inference must reuse allocated tensors
    Ort::Value input_tensor_;   // Pre-allocated
    Ort::Value output_tensor_;  // Pre-allocated
};
```

---

## 📚 References

- [Data-Oriented Design](https://www.dataorienteddesign.com/dodbook/)
- [CppCoreGuidelines](https://isocpp.github.io/CppCoreGuidelines/)
- [Agner Fog's Optimization Manuals](https://www.agner.org/optimize/)
- [Real-Time C++ Best Practices](https://www.youtube.com/watch?v=Tof5pRedskI)

---

*Last updated: January 2026*
*Maintainer: PhantomCore Team*
