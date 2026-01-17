cmake_minimum_required(VERSION 3.20)
project(PhantomCore 
    VERSION 0.1.0
    DESCRIPTION "Ultra-Low-Latency Neural Signal Processing Library"
    LANGUAGES CXX
)

# ============================================================================
# Build Configuration
# ============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific flags
if(MSVC)
    add_compile_options(/W4 /WX /arch:AVX2 /fp:fast /O2)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    add_compile_options(-march=native -ffast-math -O3)
    
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -fsanitize=address,undefined)
        add_link_options(-fsanitize=address,undefined)
    endif()
endif()

# ============================================================================
# Options
# ============================================================================
option(PHANTOMCORE_BUILD_EXAMPLES "Build example applications" ON)
option(PHANTOMCORE_BUILD_TESTS "Build unit tests" ON)
option(PHANTOMCORE_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(PHANTOMCORE_ENABLE_SIMD "Enable SIMD optimizations" ON)
option(PHANTOMCORE_ENABLE_CUDA "Enable CUDA acceleration (experimental)" OFF)

# ============================================================================
# Dependencies
# ============================================================================
include(FetchContent)

# IXWebSocket - Modern C++ WebSocket library
FetchContent_Declare(
    ixwebsocket
    GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
    GIT_TAG v11.4.4
)
FetchContent_MakeAvailable(ixwebsocket)

# msgpack-c - MessagePack for C/C++
FetchContent_Declare(
    msgpack-c
    GIT_REPOSITORY https://github.com/msgpack/msgpack-c.git
    GIT_TAG cpp-6.1.0
)
set(MSGPACK_CXX20 ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(msgpack-c)

# Eigen - Linear algebra (for Kalman filter)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# ============================================================================
# Core Library
# ============================================================================
add_library(phantomcore STATIC
    src/spike_detector.cpp
    src/kalman_decoder.cpp
    src/stream_client.cpp
    src/simd_utils.cpp
    src/ring_buffer.cpp
    src/latency_tracker.cpp
)

target_include_directories(phantomcore PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(phantomcore PUBLIC
    ixwebsocket
    msgpack-cxx
    Eigen3::Eigen
)

if(PHANTOMCORE_ENABLE_SIMD)
    target_compile_definitions(phantomcore PUBLIC PHANTOMCORE_SIMD_ENABLED)
endif()

# ============================================================================
# Examples
# ============================================================================
if(PHANTOMCORE_BUILD_EXAMPLES)
    add_executable(realtime_demo examples/realtime_demo.cpp)
    target_link_libraries(realtime_demo PRIVATE phantomcore)

    add_executable(latency_benchmark examples/latency_benchmark.cpp)
    target_link_libraries(latency_benchmark PRIVATE phantomcore)

    add_executable(spike_visualizer examples/spike_visualizer.cpp)
    target_link_libraries(spike_visualizer PRIVATE phantomcore)

    add_executable(closed_loop_sim examples/closed_loop_sim.cpp)
    target_link_libraries(closed_loop_sim PRIVATE phantomcore)
endif()

# ============================================================================
# Tests
# ============================================================================
if(PHANTOMCORE_BUILD_TESTS)
    enable_testing()
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    FetchContent_MakeAvailable(googletest)
    
    add_executable(phantomcore_tests
        tests/test_spike_detector.cpp
        tests/test_kalman_decoder.cpp
        tests/test_ring_buffer.cpp
        tests/test_simd_utils.cpp
    )
    target_link_libraries(phantomcore_tests PRIVATE phantomcore GTest::gtest_main)
    
    include(GoogleTest)
    gtest_discover_tests(phantomcore_tests)
endif()

# ============================================================================
# Benchmarks
# ============================================================================
if(PHANTOMCORE_BUILD_BENCHMARKS)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
    
    add_executable(phantomcore_benchmarks
        benchmarks/bench_spike_detector.cpp
        benchmarks/bench_kalman_decoder.cpp
        benchmarks/bench_simd.cpp
    )
    target_link_libraries(phantomcore_benchmarks PRIVATE phantomcore benchmark::benchmark)
endif()

# ============================================================================
# Install
# ============================================================================
include(GNUInstallDirs)
install(TARGETS phantomcore
    EXPORT PhantomCoreTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ============================================================================
# Summary
# ============================================================================
message(STATUS "")
message(STATUS "PhantomCore ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  SIMD:           ${PHANTOMCORE_ENABLE_SIMD}")
message(STATUS "  CUDA:           ${PHANTOMCORE_ENABLE_CUDA}")
message(STATUS "  Examples:       ${PHANTOMCORE_BUILD_EXAMPLES}")
message(STATUS "  Tests:          ${PHANTOMCORE_BUILD_TESTS}")
message(STATUS "  Benchmarks:     ${PHANTOMCORE_BUILD_BENCHMARKS}")
message(STATUS "")
