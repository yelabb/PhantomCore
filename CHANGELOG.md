# Changelog

All notable changes to PhantomCore will be documented in this file.

## [Unreleased]

### Added
- Initial project structure with CMake build system
- Core data types for neural signal processing (`types.hpp`)
- SIMD-optimized vector operations with AVX2 support (`simd_utils.hpp`)
- Lock-free ring buffer for real-time streaming (`ring_buffer.hpp`)
- High-precision latency tracking utilities (`latency_tracker.hpp`)
- Spike detection engine with adaptive thresholding (`spike_detector.hpp`)
- Simple k-means spike sorter (`SpikeSorter`)
- Kalman filter neural decoder (`kalman_decoder.hpp`)
- Linear decoder baseline implementation
- Velocity Kalman filter decoder variant
- WebSocket client for PhantomLink integration (`stream_client.hpp`)
- MessagePack packet parser
- Example applications:
  - `realtime_demo` - Live neural data streaming
  - `latency_benchmark` - Performance measurement suite
  - `spike_visualizer` - Console-based activity heatmap
  - `closed_loop_sim` - Full closed-loop BCI simulation
- Comprehensive unit tests with Google Test
- Performance benchmarks with Google Benchmark
- Full README documentation

### Technical Highlights
- Sub-millisecond end-to-end latency achieved
- AVX2 SIMD optimization for 142-channel processing
- Lock-free data structures for deterministic timing
- Eigen-based Kalman filter implementation
- Binary MessagePack protocol support

## [0.1.0] - 2026-01-17

- Initial release
