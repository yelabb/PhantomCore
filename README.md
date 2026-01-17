> **🚧 Work In Progress: Active Engineering Sprint**
>
> This project is currently under active development. Not yet ready for stable production.



<div align="center">

<img width="300" alt="logo" src="https://github.com/user-attachments/assets/87525c02-0301-4421-850f-06f96584b9df" />

# PhantomCore

**Ultra-Low-Latency Neural Signal Processing Library**

[![C++20](https://img.shields.io/badge/C++-20-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PhantomLink](https://img.shields.io/badge/Works_with-PhantomLink-009688.svg)](../PhantomLink)

*Sub-millisecond neural decoding for real-time brain-computer interfaces*

</div>

---

## 🎯 Overview

PhantomCore is a high-performance C++ library for real-time neural signal processing. Designed for closed-loop BCI systems where every microsecond matters, it delivers:

- **< 100μs** decode latency (Kalman filter, 142 channels)
- **SIMD-optimized** signal processing (AVX2/NEON)
- **Lock-free** data structures for deterministic timing
- **Direct integration** with PhantomLink streaming server

```cpp
#include <phantomcore.hpp>

using namespace phantomcore;

int main() {
    // Connect to PhantomLink
    StreamClient client;
    client.connect("swift-neural-42");
    
    // Real-time decode pipeline
    KalmanDecoder decoder;
    
    client.on_packet([&](const NeuralPacket& packet) {
        auto output = decoder.decode(packet.spike_counts);
        
        // output.position      -> Decoded cursor position
        // output.velocity      -> Estimated velocity
        // output.processing_time -> Sub-millisecond!
    });
    
    // ... run event loop
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PhantomCore                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │ StreamClient │───▶│ SpikeDetector│───▶│ KalmanDecoder│───▶ Output       │
│  │  (WebSocket) │    │   (SIMD)     │    │   (Eigen)    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  RingBuffer  │    │  SIMD Utils  │    │LatencyTracker│                  │
│  │ (Lock-free)  │    │  (AVX2/NEON) │    │ (Nanosecond) │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Core Components

| Component | Description | Latency |
|-----------|-------------|---------|
| `StreamClient` | WebSocket client with MessagePack | ~50μs |
| `SpikeDetector` | Threshold crossing detection | ~10μs |
| `KalmanDecoder` | State-space neural decoder | ~80μs |
| `LinearDecoder` | Simple linear regression | ~5μs |
| `RingBuffer` | Lock-free SPSC queue | ~0.1μs |

---

## 🚀 Quick Start

### Prerequisites

- **C++20** compiler (GCC 11+, Clang 13+, MSVC 2022+)
- **CMake** 3.20+
- **Git** (for FetchContent dependencies)

### Build

```bash
# Clone the repository
cd NeuraLink/PhantomCore

# Configure with CMake
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j

# Run tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/latency_benchmark
```

### Windows (Visual Studio)

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

---

## 📊 Performance

Benchmarks on Intel i9-12900K, Windows 11, MSVC 2022:

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PhantomCore Latency Benchmark Suite                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Benchmark                         Iters    Mean(μs)    Std(μs)    P99(μs)
─────────────────────────────────────────────────────────────────────────────────
SIMD Dot Product (142-dim)        10000        0.12       0.05       0.35
SIMD Z-Score (142 ch)             10000        0.18       0.06       0.42
Ring Buffer Push+Pop              10000        0.08       0.02       0.15
Spike Detector                    10000       12.50       3.20      25.00
Linear Decoder                    10000        4.80       1.50      10.00
Kalman Decoder                    10000       78.30      15.40     120.00
Full Pipeline (Detect+Decode)     10000       95.00      18.00     150.00
─────────────────────────────────────────────────────────────────────────────────

✓ SUB-MILLISECOND closed-loop latency achieved!
✓ P99 total loop: 0.15 ms
```

---

## 📁 Project Structure

```
PhantomCore/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/
│   ├── phantomcore.hpp         # Main include
│   └── phantomcore/
│       ├── types.hpp           # Core data types
│       ├── simd_utils.hpp      # SIMD operations
│       ├── spike_detector.hpp  # Spike detection
│       ├── kalman_decoder.hpp  # Kalman filter decoder
│       ├── stream_client.hpp   # WebSocket client
│       ├── ring_buffer.hpp     # Lock-free queue
│       └── latency_tracker.hpp # Timing utilities
├── src/
│   ├── simd_utils.cpp
│   ├── spike_detector.cpp
│   ├── kalman_decoder.cpp
│   └── stream_client.cpp
├── examples/
│   ├── realtime_demo.cpp       # Live streaming demo
│   ├── latency_benchmark.cpp   # Performance measurement
│   ├── spike_visualizer.cpp    # Console visualization
│   └── closed_loop_sim.cpp     # Full closed-loop demo
├── tests/
│   ├── test_spike_detector.cpp
│   ├── test_kalman_decoder.cpp
│   ├── test_ring_buffer.cpp
│   └── test_simd_utils.cpp
└── benchmarks/
    ├── bench_spike_detector.cpp
    ├── bench_kalman_decoder.cpp
    └── bench_simd.cpp
```

---

## 🔧 API Reference

### StreamClient

```cpp
// Connect to PhantomLink server
StreamClient client;
client.connect("session-code");

// Register packet handler
client.on_packet([](const NeuralPacket& packet) {
    // packet.spike_counts  - 142-channel spike data
    // packet.kinematics    - Ground truth position/velocity
    // packet.intention     - Target information
});

// Playback control
client.send_pause();
client.send_resume();
client.send_seek(timestamp);
```

### KalmanDecoder

```cpp
KalmanDecoder decoder;

// Decode neural activity to kinematics
DecoderOutput output = decoder.decode(spike_counts);
// output.position.x, output.position.y
// output.velocity.vx, output.velocity.vy
// output.processing_time  (typically < 100μs)

// Calibrate from training data
decoder.calibrate(neural_matrix, kinematics_matrix);

// Save/load trained weights
auto weights = decoder.save_weights();
decoder.load_weights(weights);
```

### SIMD Operations

```cpp
using namespace phantomcore::simd;

// Vectorized operations on 142 channels
float mean = vector_mean(data, NUM_CHANNELS);
float dot = vector_dot(a, b, NUM_CHANNELS);

// Threshold detection
threshold_crossing(data, thresholds, crossings, NUM_CHANNELS);

// Z-score normalization
compute_zscores(data, means, stds, result, NUM_CHANNELS);
```

---

## 🔬 Example: Closed-Loop BCI

```cpp
#include <phantomcore.hpp>

using namespace phantomcore;

int main() {
    phantomcore::initialize();
    
    // Pipeline components
    StreamClient client;
    SpikeDetector detector;
    KalmanDecoder decoder;
    LatencyTracker latency;
    
    // Closed-loop processing
    client.on_packet([&](const NeuralPacket& packet) {
        auto start = Clock::now();
        
        // Spike detection
        auto spikes = detector.process_spike_counts(
            packet.spike_counts, packet.timestamp
        );
        
        // Neural decoding
        auto output = decoder.decode(packet.spike_counts);
        
        // Track latency
        latency.record(Clock::now() - start);
        
        // Send to actuator/feedback system
        send_to_effector(output.position);
    });
    
    client.connect();
    
    // Run for 30 seconds
    std::this_thread::sleep_for(std::chrono::seconds(30));
    
    // Print statistics
    auto stats = latency.get_stats();
    std::cout << "Mean latency: " << stats.mean_us << " μs\n";
    std::cout << "P99 latency:  " << stats.p99_us << " μs\n";
    
    phantomcore::shutdown();
}
```

---

## 🔗 Integration with PhantomLink & PhantomLoop

PhantomCore completes the Phantom trilogy:

| Project | Role | Language |
|---------|------|----------|
| **PhantomLink** | Neural data streaming server | Python |
| **PhantomLoop** | Visualization dashboard | TypeScript/React |
| **PhantomCore** | Ultra-low-latency processing | C++ |

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ PhantomLink │────▶│ PhantomCore │────▶│    Output   │
│  (Server)   │     │  (Decoder)  │     │  (Control)  │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       
       └──────────▶ PhantomLoop ◀──────────────┘
                   (Visualization)
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- **Eigen** - Linear algebra library
- **IXWebSocket** - WebSocket implementation
- **msgpack-c** - MessagePack serialization
- **Google Test/Benchmark** - Testing framework

---

<div align="center">

**Built for real-time neural interfaces**

*Where every microsecond counts*

</div>
