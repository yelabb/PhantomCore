#include "phantomcore/simd_utils.hpp"
#include <atomic>

namespace phantomcore {

static std::atomic<bool> g_initialized{false};

void initialize() {
    if (g_initialized.exchange(true)) {
        return; // Already initialized
    }
    
    // Detect SIMD capabilities
    simd::has_avx2();  // Will cache the result
    simd::has_neon();
}

void shutdown() {
    g_initialized.store(false);
}

const char* build_info() {
    static const char* info = 
        "PhantomCore v0.1.0\n"
        "Compiled: " __DATE__ " " __TIME__ "\n"
#ifdef PHANTOMCORE_SIMD_ENABLED
        "SIMD: Enabled\n"
#ifdef PHANTOMCORE_HAS_AVX2
        "AVX2: Yes\n"
#elif defined(PHANTOMCORE_HAS_NEON)
        "NEON: Yes\n"
#else
        "SIMD Backend: Scalar fallback\n"
#endif
#else
        "SIMD: Disabled\n"
#endif
        "Compiler: "
#ifdef _MSC_VER
        "MSVC " 
#elif defined(__clang__)
        "Clang "
#elif defined(__GNUC__)
        "GCC "
#else
        "Unknown "
#endif
        ;
    return info;
}

} // namespace phantomcore
