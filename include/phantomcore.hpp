#pragma once

// PhantomCore - Ultra-Low-Latency Neural Signal Processing Library
// Version 0.1.0

#include "phantomcore/types.hpp"
#include "phantomcore/aligned_allocator.hpp"
#include "phantomcore/simd_utils.hpp"
#include "phantomcore/bandpass_filter.hpp"
#include "phantomcore/dimensionality_reduction.hpp"
#include "phantomcore/regularization.hpp"
#include "phantomcore/spike_detector.hpp"
#include "phantomcore/kalman_decoder.hpp"
#include "phantomcore/model_checkpoint.hpp"
#include "phantomcore/stream_client.hpp"
#include "phantomcore/ring_buffer.hpp"
#include "phantomcore/latency_tracker.hpp"

namespace phantomcore {

/// Library version
constexpr const char* VERSION = "0.1.0";

/// Library version components
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

/**
 * @brief Initialize the library
 * Call once at startup. Detects SIMD capabilities and
 * initializes any global resources.
 */
void initialize();

/**
 * @brief Shutdown the library
 * Call before program exit to clean up resources.
 */
void shutdown();

/**
 * @brief Get build information string
 */
const char* build_info();

}  // namespace phantomcore
