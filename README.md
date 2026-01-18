> **🚧 Work In Progress: Active Engineering Sprint**
>
> This project is currently under active development. Not yet ready for stable production.



<div align="center">

<img width="300" alt="logo" src="https://github.com/user-attachments/assets/87525c02-0301-4421-850f-06f96584b9df" />

# PhantomCore

**Ultra-Low-Latency Neural Signal Processing Library**

[![CI](https://github.com/yelabb/PhantomCore/actions/workflows/ci.yml/badge.svg)](https://github.com/yelabb/PhantomCore/actions/workflows/ci.yml)
[![C++23](https://img.shields.io/badge/C++-23-blue.svg)](https://isocpp.org/)
[![CMake](https://img.shields.io/badge/CMake-3.20+-green.svg)](https://cmake.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PhantomLink](https://img.shields.io/badge/Works_with-PhantomLink-009688.svg)](https://github.com/yelabb/PhantomLink)

*Sub-millisecond neural decoding for real-time brain-computer interfaces*

</div>

---

## 🎯 Overview

PhantomCore is a high-performance C++ library for real-time neural signal processing. Designed for closed-loop BCI systems where every microsecond matters, it delivers:

- **< 15μs** computational latency (spike detection + Kalman decode, see [benchmark notes](#-performance))
- **~4μs** Kalman decoder (Woodbury-optimized, 142 channels → 2D cursor)
- **SIMD-optimized** signal processing (AVX2/NEON)
- **Lock-free** data structures for deterministic timing
- **Direct integration** with PhantomLink streaming server

> **Note**: PhantomCore is a *signal processing library*, not a complete BCI stack. Hardware drivers and data acquisition require external tools. See [Scope & Limitations](#%EF%B8%8F-scope--limitations).

### Advanced Capabilities

| Feature | Description | Benefit |
|---------|-------------|---------|
| **🚀 GPU Acceleration** | CUDA-based multi-probe decoding | Scale to 4000+ channels |
| **🔄 Adaptive Calibration** | Online RLS with drift detection | Zero-downtime recalibration |
| **🧠 Neural Network Decoder** | ONNX Runtime inference | +30-50% accuracy over Kalman |

```cpp
#include <phantomcore.hpp>

using namespace phantomcore;

int main() {
    // Configure for your hardware (runtime - no recompilation needed!)
    auto config = ChannelConfig::neuropixels();  // 384 channels
    // Or: ChannelConfig::utah_array_96()        // 96 channels
    // Or: ChannelConfig::mc_maze()              // 142 channels (default)
    // Or: ChannelConfig::custom(256, "MyArray") // Custom hardware
    
    // Connect to PhantomLink
    StreamClient client;
    client.connect("swift-neural-42");
    
    // Real-time decode pipeline with dynamic channels
    KalmanDecoder::Config decoder_config;
    decoder_config.channel_config = config;
    KalmanDecoder decoder(decoder_config);
    
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

| Component | Description | Mean Latency |
|-----------|-------------|--------------|
| `StreamClient` | WebSocket client for PhantomLink | ~10μs |
| `SpikeDetector` | Threshold crossing + bandpass filtering | ~13μs |
| `KalmanDecoder` | Woodbury-optimized state-space decoder | ~4μs |
| `LinearDecoder` | Simple linear regression | ~0.5μs |
| `RingBuffer` | Lock-free SPSC queue | ~0.05μs |
| `PCAProjector` | Dimensionality reduction (142→15 dims) | ~2μs |
| `RidgeRegression` | Regularized calibration | - |

### Advanced Decoders

| Component | Description | Use Case |
|-----------|-------------|----------|
| `GPUDecoder` | CUDA-accelerated Kalman for high channel counts | Neuropixels 2.0 (960+ ch) |
| `MultiProbeDecoder` | Coordinates multiple probes with fusion | 4000+ channel systems |
| `AdaptiveDecoder` | Online RLS calibration with drift detection | Long-term implants |
| `NeuralNetDecoder` | ONNX Runtime inference (MLP/LSTM/TCN) | +30-50% accuracy |

---

## 🆕 Advanced Features

### GPU Acceleration (CUDA)

Scale to thousands of channels with GPU-accelerated decoding:

```cpp
#include <phantomcore.hpp>
using namespace phantomcore;

// Check GPU availability
if (GPUDecoder::is_gpu_available()) {
    GPUDecoder::Config config;
    config.channel_config = ChannelConfig::neuropixels_2();  // 960 channels
    config.execution_mode = GPUExecutionMode::Pipelined;     // Lowest latency
    
    GPUDecoder decoder(config);
    
    // Same API as KalmanDecoder
    auto result = decoder.decode(spike_data);
    if (result) {
        std::cout << "Position: " << result->position.x << ", " << result->position.y << "\n";
    }
    
    // Performance monitoring
    auto stats = decoder.get_stats();
    std::cout << "Kernel latency: " << stats.kernel_latency.mean_us << " μs\n";
}
```

**Multi-probe support** for next-generation implants:

```cpp
MultiProbeDecoder::Config config;
config.probes = {
    {ChannelConfig::neuropixels(), "Motor Cortex"},
    {ChannelConfig::neuropixels(), "Premotor Cortex"},
    {ChannelConfig::neuropixels(), "Parietal Cortex"},
    {ChannelConfig::neuropixels(), "Somatosensory"}
};
config.fusion = MultiProbeDecoder::Config::FusionStrategy::KalmanFusion;

MultiProbeDecoder decoder(config);
// Decode from 4 probes simultaneously (1536 channels)
auto output = decoder.decode(probe_data_vector);
```

### Adaptive Online Calibration

Compensate for neural drift without stopping the experiment:

```cpp
AdaptiveDecoder::Config config;
config.adaptive.learning_rate = 0.01f;
config.adaptive.forgetting_factor = 0.995f;  // Prioritize recent data
config.adaptive.drift_method = DriftDetectionMethod::HybridMultiMetric;

AdaptiveDecoder decoder(config);

// Initial calibration
decoder.calibrate(training_neural, training_kinematics);

// Closed-loop with online adaptation
client.on_packet([&](const NeuralPacket& packet) {
    auto output = decoder.decode_and_update(
        packet.spike_counts,
        {packet.kinematics.position.x, packet.kinematics.position.y,
         packet.kinematics.velocity.vx, packet.kinematics.velocity.vy}
    );
    
    // Check for drift
    auto drift = decoder.get_drift_status();
    if (drift.drift_detected) {
        std::cout << "Drift detected! Score: " << drift.drift_score << "\n";
        if (drift.recommended_action == DriftDetectionResult::Action::Recalibrate) {
            // Trigger recalibration UI
        }
    }
});

// Rollback if adaptation went wrong
decoder.rollback(5);  // Undo last 5 updates
```

### Neural Network Decoder (ONNX)

Achieve higher accuracy with deep learning models:

```cpp
NeuralNetDecoder::Config config;
config.channel_config = ChannelConfig::mc_maze();
config.backend = NNBackend::ONNX;
config.use_gpu = true;
config.hybrid_mode = true;  // NN + Kalman fusion for stability

NeuralNetDecoder decoder(config);

// Load trained model
auto result = decoder.load_model("models/tcn_decoder.onnx");
if (!result) {
    std::cerr << "Failed to load model: " << nn_error_string(result.error()) << "\n";
    return 1;
}

// Warm up for consistent latency
decoder.warmup(10);

// Decode with hybrid NN + Kalman
auto output = decoder.decode(spike_data);
if (output) {
    std::cout << "Decoded position: " << output->position.x << ", " << output->position.y << "\n";
}

// Adjust NN vs Kalman blend
decoder.set_nn_weight(0.8f);  // 80% NN, 20% Kalman

// Benchmark
auto latency = decoder.benchmark(100);
std::cout << "Mean inference: " << latency.mean_us << " μs\n";
```

**Generate training scripts** for use with PhantomLink:

```cpp
auto architecture = NNModelBuilder::tcn({64, 64, 32}, 3);
std::string script = NNModelBuilder::generate_training_script(
    architecture, 142, 4, "my_decoder"
);
// Save and run in Python to train model
std::ofstream("train_model.py") << script;
```

---

## ⚡ Hardware Flexibility

PhantomCore supports **runtime channel configuration** - switch between different neural recording hardware without recompilation:

```cpp
// Pre-defined hardware presets
auto utah96  = ChannelConfig::utah_array_96();   // 96 channels
auto utah128 = ChannelConfig::utah_array_128();  // 128 channels
auto mcmaze  = ChannelConfig::mc_maze();         // 142 channels (default)
auto npx1    = ChannelConfig::neuropixels();     // 384 channels
auto npx2    = ChannelConfig::neuropixels_2();   // 960 channels

// Custom hardware
auto custom  = ChannelConfig::custom(256, "Custom Array");

// All components accept ChannelConfig
SpikeDetector detector(config);
KalmanDecoder decoder(config);
LinearDecoder linear(config);
```

| Hardware Preset | Channels | Use Case |
|-----------------|----------|----------|
| `UtahArray96` | 96 | Utah microelectrode array |
| `UtahArray128` | 128 | High-density Utah array |
| `MCMaze142` | 142 | MC_Maze benchmark dataset |
| `Neuropixels384` | 384 | Neuropixels 1.0 probe |
| `Neuropixels960` | 960 | Neuropixels 2.0 probe |
| `Custom` | Any | User-defined hardware |

---

## 🚀 Quick Start

### Prerequisites

- **C++23** compiler (GCC 13+, Clang 17+, MSVC 2022+)
- **CMake** 3.20+
- **Git** (for FetchContent dependencies)

### Build

```bash
# Clone the repository
cd PhantomCore

# Configure with CMake (core library)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build --config Release -j

# Run tests
ctest --test-dir build --output-on-failure

# Run benchmarks
./build/latency_benchmark
```

### Optional Features

Enable advanced decoders with optional dependencies:

```bash
# Enable GPU acceleration (requires CUDA Toolkit 11.0+)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPHANTOMCORE_ENABLE_CUDA=ON

# Enable neural network decoder (requires ONNX Runtime)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPHANTOMCORE_ENABLE_ONNX=ON

# Enable all features
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DPHANTOMCORE_ENABLE_CUDA=ON \
      -DPHANTOMCORE_ENABLE_ONNX=ON
```

| Build Option | Dependency | Description |
|--------------|------------|-------------|
| `PHANTOMCORE_ENABLE_CUDA` | CUDA Toolkit 11.0+ | GPUDecoder, MultiProbeDecoder |
| `PHANTOMCORE_ENABLE_ONNX` | ONNX Runtime 1.16+ | NeuralNetDecoder |

### Windows (Visual Studio)

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release

# With CUDA support
cmake -B build -G "Visual Studio 17 2022" -A x64 -DPHANTOMCORE_ENABLE_CUDA=ON
```

---

## 📊 Performance

**Actual benchmarks** on a quite old 1st generation of Intel Core i7, Windows 10, MSVC 2022, AVX2 enabled:

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PhantomCore Latency Benchmark Suite                        ║
║                    Sub-Millisecond Neural Processing                          ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Build: PhantomCore v0.1.0
SIMD: AVX2 (256-bit)

Benchmark                         Iters    Mean(μs)    Std(μs)    P99(μs)    Max(μs)
─────────────────────────────────────────────────────────────────────────────────────
SIMD Dot Product (142-dim)        10000        0.06       0.11       0.10       8.10
SIMD Z-Score (142 ch)             10000        0.15       2.39       0.20     236.40
Ring Buffer Push+Pop              10000        0.05       0.05       0.10       0.30
Spike Detector                    10000       12.92     130.16      64.80    8210.40
Linear Decoder                    10000        0.51       4.24       0.60     414.70
Kalman Decoder                    10000        3.99      27.59      33.10    2189.70
Full Pipeline (Detect+Decode)     10000       14.39      97.96      36.00    7866.80
─────────────────────────────────────────────────────────────────────────────────────

Summary:
  Full Pipeline Mean:    14.39 μs ✓ SUB-MILLISECOND
  Full Pipeline P99:     36.00 μs ✓ SUB-MILLISECOND
  Throughput:            69,509 packets/sec
  Real-time Headroom:    1738x (at 40Hz streaming)
```

> ⚠️ **Benchmark Methodology Notes**
>
> These benchmarks measure **computational latency only** under ideal conditions:
> - Input is pre-binned spike counts (not raw 30kHz samples)
> - Does not include I/O, network transport, or hardware acquisition time
> - Cache is warmed before measurement (100 iterations)
> - No OS scheduling jitter or interrupt handling
> - Real-world latency will be higher and depends on your hardware, system load, and data acquisition pipeline
>
> For end-to-end latency including PhantomLink network transport, expect an additional 100-500μs depending on network conditions.

### Key Optimizations

- **Kalman Decoder**: Uses Woodbury matrix identity for 4×4 inversion instead of N×N
- **PCA Latent Space**: Optional dimensionality reduction (142→15 dims) for faster updates
- **Ridge Regression**: L2-regularized calibration prevents overfitting on noisy neural data
- **Bandpass Filtering**: 300-3000Hz IIR filter for spike isolation
- **SIMD**: AVX2 vectorized spike z-score normalization and dot products
- **Eigen Vectorization**: Matrix-vector products auto-vectorized with `-march=native -O3` (GCC/Clang) or `/arch:AVX2 /O2` (MSVC)
- **Aligned Memory**: 32-byte aligned allocators for safe SIMD operations
- **Lock-free**: Ring buffer with atomic operations for deterministic timing
- **Dynamic Channels**: Runtime hardware configuration without recompilation
- **Full Serialization**: `ModelCheckpoint` saves complete pipeline state (not just weights)

---

## 📁 Project Structure

```
PhantomCore/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── include/
│   ├── phantomcore.hpp         # Main include
│   └── phantomcore/
│       ├── types.hpp           # Core data types + ChannelConfig
│       ├── simd_utils.hpp      # SIMD operations
│       ├── spike_detector.hpp  # Spike detection + bandpass
│       ├── bandpass_filter.hpp # IIR filtering (300-3000Hz)
│       ├── kalman_decoder.hpp  # Kalman filter decoder
│       ├── gpu_decoder.hpp     # GPU-accelerated decoder (CUDA)
│       ├── adaptive_decoder.hpp    # Online adaptive calibration
│       ├── neural_net_decoder.hpp  # ONNX neural network decoder
│       ├── dimensionality_reduction.hpp  # PCA projector
│       ├── regularization.hpp  # Ridge/ElasticNet regression
│       ├── aligned_allocator.hpp  # SIMD-safe memory
│       ├── stream_client.hpp   # WebSocket client
│       ├── ring_buffer.hpp     # Lock-free queue
│       └── latency_tracker.hpp # Timing utilities
├── src/
│   ├── simd_utils.cpp
│   ├── spike_detector.cpp
│   ├── kalman_decoder.cpp
│   ├── gpu_decoder.cpp         # CUDA implementation
│   ├── adaptive_decoder.cpp    # RLS online learning
│   ├── neural_net_decoder.cpp  # ONNX Runtime inference
│   ├── dimensionality_reduction.cpp
│   ├── regularization.cpp
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
├── benchmarks/
│   ├── bench_spike_detector.cpp
│   ├── bench_kalman_decoder.cpp
│   └── bench_simd.cpp
└── docs/
    └── CODING_PRINCIPLES.md    # Engineering guidelines
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
// Configure for your hardware
KalmanDecoder::Config config;
config.channel_config = ChannelConfig::neuropixels();  // 384 channels
config.use_latent_space = true;   // Enable PCA for faster updates
config.pca_components = 15;       // Reduce to 15 latent dims
config.ridge_lambda = 1e-3f;      // Regularization strength

KalmanDecoder decoder(config);

// Decode neural activity to kinematics
SpikeData spikes(config.channel_config);
// ... fill spikes ...
DecoderOutput output = decoder.decode(spikes);
// output.position.x, output.position.y
// output.velocity.vx, output.velocity.vy
// output.processing_time  (typically < 100μs)

// Calibrate from training data (with Ridge regularization)
decoder.calibrate(neural_matrix, kinematics_matrix);
```

### ModelCheckpoint (Session Persistence)

```cpp
// After calibration, save the COMPLETE pipeline state
ModelCheckpoint checkpoint = create_checkpoint(decoder);
checkpoint.model_name = "Subject01_Session03";
checkpoint.notes = "Trained on 5000 trials, R²=0.87";

// Save to disk (binary format)
checkpoint.save("models/subject01_session03.phmc");

// Later: restore a session
ModelCheckpoint loaded = ModelCheckpoint::load("models/subject01_session03.phmc");
if (loaded.validate()) {
    std::cout << "Loaded model: " << loaded.model_name << "\n";
    std::cout << "Channels: " << loaded.channel_config.num_channels << "\n";
    std::cout << "R² score: " << loaded.calibration_r2_score << "\n";
    
    // Create decoder with same config
    KalmanDecoder restored_decoder(loaded.channel_config);
    restore_from_checkpoint(restored_decoder, loaded);
    
    // Ready to decode!
}
```

**Checkpoint includes:**
- Channel configuration (hardware preset)
- Spike normalization (mean/std for z-score)
- PCA projection matrix and centering
- Kalman observation matrix (H)
- Latent observation matrix (H_latent)
- Process/measurement noise parameters
- Calibration metadata (R², λ, sample count)

### SIMD Operations

```cpp
using namespace phantomcore::simd;

// Dynamic channel operations (runtime size)
SpikeData data(ChannelConfig::neuropixels());  // 384 channels
float mean = ChannelProcessor::compute_mean_rate(data);

// Span-based API for flexibility
std::span<float> rates = data.span();
ChannelProcessor::compute_zscores(rates, means, stds, result);

// Vectorized operations (any size)
float dot = vector_dot(a.data(), b.data(), a.size());

// Threshold detection
threshold_crossing(data, thresholds, crossings, num_channels);
```

### SpikeDetector with Bandpass Filtering

```cpp
// Configure detector for your hardware
SpikeDetector::Config config;
config.bandpass.low_cutoff_hz = 300.0f;   // High-pass for LFP rejection
config.bandpass.high_cutoff_hz = 3000.0f; // Low-pass for noise
config.threshold_multiplier = -4.5f;       // Detection threshold

SpikeDetector detector(ChannelConfig::utah_array_96(), config);

// Process raw neural samples
auto events = detector.process_batch(samples, batch_size, 96, timestamp, 30000.0);
```

### PCA Dimensionality Reduction

```cpp
PCAProjector::Config pca_config;
pca_config.n_components = 15;          // Target latent dims
pca_config.variance_threshold = 0.95f; // Or use variance explained

PCAProjector pca(pca_config);
pca.fit(training_data);  // [n_samples x n_channels]

// Transform new data (142 → 15 dims)
Eigen::VectorXf latent = pca.transform(spike_vector);

std::cout << "Variance explained: " << pca.cumulative_variance_explained() << "\n";
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

## ⚠️ Scope & Limitations

PhantomCore is a **signal processing library**, not a complete BCI system. Understanding its scope helps set realistic expectations:

### What PhantomCore IS

| Capability | Description |
|------------|-------------|
| ✅ Signal Processing | Bandpass filtering, spike detection, feature extraction |
| ✅ Neural Decoding | Kalman filters, linear decoders, neural network inference |
| ✅ Low-Latency Math | SIMD-optimized matrix operations, lock-free buffers |
| ✅ Network Client | WebSocket client for PhantomLink data streams |

### What PhantomCore is NOT

| Not Included | You Need |
|--------------|----------|
| ❌ Hardware drivers | [SpikeGLX](https://billkarsh.github.io/SpikeGLX/), [Open Ephys](https://open-ephys.org/), or vendor SDKs (Intan, Ripple, Blackrock) |
| ❌ Raw data acquisition | Hardware-specific APIs for Neuropixels, Utah arrays, etc. |
| ❌ Electrical safety | Medical device certification, isolation, grounding |
| ❌ Clinical validation | FDA/CE approval for human use |

### Typical Integration

```
┌──────────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│ Recording System │────▶│  SpikeGLX /  │────▶│ PhantomLink │────▶│ Phantom- │
│ (Neuropixels,    │     │  Open Ephys  │     │  (Stream)   │     │   Core   │
│  Utah Array)     │     │  (Acquire)   │     └─────────────┘     └──────────┘
└──────────────────┘     └──────────────┘
        ▲                                                              │
        │                                                              ▼
   HARDWARE LAYER                                              ┌──────────────┐
   (Not PhantomCore)                                           │   Actuator   │
                                                               │  (Prosthetic,│
                                                               │   Cursor)    │
                                                               └──────────────┘
```

PhantomCore assumes you already have a working data acquisition pipeline. It focuses exclusively on **sub-millisecond processing** of neural data, leaving hardware integration to specialized tools.

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

