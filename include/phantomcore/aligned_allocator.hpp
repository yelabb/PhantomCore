#pragma once

#include <cstdlib>
#include <memory>
#include <vector>
#include <array>
#include <new>
#include <type_traits>

#ifdef _MSC_VER
#include <malloc.h>  // _aligned_malloc/_aligned_free
#endif

namespace phantomcore {

// ============================================================================
// Aligned Memory Allocation
// ============================================================================

/**
 * @brief Aligned memory allocation/deallocation functions
 * 
 * Cross-platform aligned memory management for SIMD operations.
 * AVX2 requires 32-byte alignment, AVX-512 requires 64-byte.
 */
namespace aligned {

/// Default alignment for AVX2 (256-bit = 32 bytes)
inline constexpr std::size_t AVX2_ALIGNMENT = 32;

/// Alignment for AVX-512 (512-bit = 64 bytes)  
inline constexpr std::size_t AVX512_ALIGNMENT = 64;

/// Cache line alignment (typically 64 bytes)
inline constexpr std::size_t CACHE_LINE = 64;

/**
 * @brief Allocate aligned memory
 * @param size Number of bytes to allocate
 * @param alignment Alignment requirement (must be power of 2)
 * @return Pointer to aligned memory, or nullptr on failure
 */
[[nodiscard]] inline void* allocate(std::size_t size, std::size_t alignment) noexcept {
    if (size == 0) return nullptr;
    
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) != 0) {
        return nullptr;
    }
    return ptr;
#endif
}

/**
 * @brief Deallocate aligned memory
 * @param ptr Pointer previously returned by aligned::allocate
 */
inline void deallocate(void* ptr) noexcept {
    if (ptr) {
#ifdef _MSC_VER
        _aligned_free(ptr);
#else
        std::free(ptr);
#endif
    }
}

/**
 * @brief Check if pointer is aligned to given boundary
 */
[[nodiscard]] inline bool is_aligned(const void* ptr, std::size_t alignment) noexcept {
    return (reinterpret_cast<std::uintptr_t>(ptr) & (alignment - 1)) == 0;
}

/**
 * @brief Assert alignment in debug builds
 */
inline void assert_aligned([[maybe_unused]] const void* ptr, 
                           [[maybe_unused]] std::size_t alignment,
                           [[maybe_unused]] const char* context = nullptr) {
#ifndef NDEBUG
    if (!is_aligned(ptr, alignment)) {
        const char* ctx = context ? context : "unknown";
        std::fprintf(stderr, 
            "[PhantomCore] ALIGNMENT ERROR: Pointer %p is not %zu-byte aligned in %s\n",
            ptr, alignment, ctx);
        std::abort();
    }
#endif
}

} // namespace aligned

// ============================================================================
// STL-Compatible Aligned Allocator
// ============================================================================

/**
 * @brief Custom allocator for aligned STL containers
 * 
 * Usage:
 *   std::vector<float, AlignedAllocator<float, 32>> aligned_vec;
 *   aligned_vec.resize(1024);  // Guaranteed 32-byte aligned
 * 
 * Critical for SIMD operations where _mm256_load_ps requires 32-byte alignment.
 * Using this allocator allows safe use of aligned load/store instructions
 * instead of unaligned variants, gaining ~3-5 cycles per operation.
 * 
 * @tparam T Element type
 * @tparam Alignment Alignment in bytes (default: 32 for AVX2)
 */
template<typename T, std::size_t Alignment = aligned::AVX2_ALIGNMENT>
class AlignedAllocator {
public:
    static_assert(Alignment >= alignof(T), "Alignment must be >= natural alignment of T");
    static_assert((Alignment & (Alignment - 1)) == 0, "Alignment must be power of 2");
    
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using propagate_on_container_move_assignment = std::true_type;
    using is_always_equal = std::true_type;
    
    constexpr AlignedAllocator() noexcept = default;
    constexpr AlignedAllocator(const AlignedAllocator&) noexcept = default;
    
    template<typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}
    
    [[nodiscard]] T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        
        // Check for overflow
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            throw std::bad_array_new_length();
        }
        
        void* ptr = aligned::allocate(n * sizeof(T), Alignment);
        if (!ptr) {
            throw std::bad_alloc();
        }
        
        return static_cast<T*>(ptr);
    }
    
    void deallocate(T* ptr, [[maybe_unused]] std::size_t n) noexcept {
        aligned::deallocate(ptr);
    }
    
    // Rebind for container internals
    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };
};

template<typename T, typename U, std::size_t A>
[[nodiscard]] constexpr bool operator==(const AlignedAllocator<T, A>&, 
                                         const AlignedAllocator<U, A>&) noexcept {
    return true;
}

template<typename T, typename U, std::size_t A>
[[nodiscard]] constexpr bool operator!=(const AlignedAllocator<T, A>&, 
                                         const AlignedAllocator<U, A>&) noexcept {
    return false;
}

// ============================================================================
// Aligned Container Type Aliases
// ============================================================================

/// Aligned vector for AVX2 SIMD operations (32-byte aligned)
template<typename T>
using AlignedVector = std::vector<T, AlignedAllocator<T, aligned::AVX2_ALIGNMENT>>;

/// Aligned vector for AVX-512 operations (64-byte aligned)
template<typename T>
using AlignedVector64 = std::vector<T, AlignedAllocator<T, aligned::AVX512_ALIGNMENT>>;

/// Cache-line aligned vector (for avoiding false sharing)
template<typename T>
using CacheAlignedVector = std::vector<T, AlignedAllocator<T, aligned::CACHE_LINE>>;

// ============================================================================
// Aligned Buffer Wrapper
// ============================================================================

/**
 * @brief Fixed-size aligned buffer for stack/member allocation
 * 
 * Alternative to std::array with guaranteed alignment.
 * Useful for fixed-size buffers in hot paths.
 * 
 * @tparam T Element type
 * @tparam N Number of elements
 * @tparam Alignment Alignment in bytes
 */
template<typename T, std::size_t N, std::size_t Alignment = aligned::AVX2_ALIGNMENT>
class alignas(Alignment) AlignedBuffer {
public:
    static_assert(N > 0, "Buffer size must be > 0");
    
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;
    using iterator = T*;
    using const_iterator = const T*;
    
    AlignedBuffer() = default;
    
    // Initialize all elements to value
    explicit AlignedBuffer(const T& value) {
        std::fill(begin(), end(), value);
    }
    
    // Initializer list
    AlignedBuffer(std::initializer_list<T> init) {
        auto it = init.begin();
        for (std::size_t i = 0; i < N && it != init.end(); ++i, ++it) {
            data_[i] = *it;
        }
    }
    
    // Element access
    [[nodiscard]] reference operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const_reference operator[](std::size_t i) const noexcept { return data_[i]; }
    
    [[nodiscard]] reference at(std::size_t i) {
        if (i >= N) throw std::out_of_range("AlignedBuffer::at");
        return data_[i];
    }
    [[nodiscard]] const_reference at(std::size_t i) const {
        if (i >= N) throw std::out_of_range("AlignedBuffer::at");
        return data_[i];
    }
    
    [[nodiscard]] pointer data() noexcept { return data_; }
    [[nodiscard]] const_pointer data() const noexcept { return data_; }
    
    [[nodiscard]] reference front() noexcept { return data_[0]; }
    [[nodiscard]] const_reference front() const noexcept { return data_[0]; }
    
    [[nodiscard]] reference back() noexcept { return data_[N - 1]; }
    [[nodiscard]] const_reference back() const noexcept { return data_[N - 1]; }
    
    // Iterators
    [[nodiscard]] iterator begin() noexcept { return data_; }
    [[nodiscard]] const_iterator begin() const noexcept { return data_; }
    [[nodiscard]] const_iterator cbegin() const noexcept { return data_; }
    
    [[nodiscard]] iterator end() noexcept { return data_ + N; }
    [[nodiscard]] const_iterator end() const noexcept { return data_ + N; }
    [[nodiscard]] const_iterator cend() const noexcept { return data_ + N; }
    
    // Capacity
    [[nodiscard]] constexpr std::size_t size() const noexcept { return N; }
    [[nodiscard]] constexpr bool empty() const noexcept { return false; }
    [[nodiscard]] constexpr std::size_t max_size() const noexcept { return N; }
    
    // Operations
    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }
    
    void swap(AlignedBuffer& other) noexcept(std::is_nothrow_swappable_v<T>) {
        std::swap_ranges(begin(), end(), other.begin());
    }
    
    // Alignment check
    [[nodiscard]] bool is_aligned() const noexcept {
        return aligned::is_aligned(data_, Alignment);
    }
    
private:
    T data_[N];
};

// ============================================================================
// SIMD-Safe Span Wrapper
// ============================================================================

/**
 * @brief Span wrapper with alignment verification
 * 
 * Wraps a pointer + size with runtime alignment check in debug builds.
 * Use this at API boundaries to catch misaligned data early.
 */
template<typename T, std::size_t Alignment = aligned::AVX2_ALIGNMENT>
class AlignedSpan {
public:
    using element_type = T;
    using value_type = std::remove_cv_t<T>;
    using size_type = std::size_t;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using iterator = T*;
    using const_iterator = const T*;
    
    constexpr AlignedSpan() noexcept : data_(nullptr), size_(0) {}
    
    AlignedSpan(T* data, std::size_t size, const char* context = nullptr) 
        : data_(data), size_(size) {
        aligned::assert_aligned(data, Alignment, context);
    }
    
    template<std::size_t N>
    AlignedSpan(AlignedBuffer<std::remove_const_t<T>, N, Alignment>& buf) noexcept
        : data_(buf.data()), size_(N) {}
    
    template<std::size_t N>
    AlignedSpan(const AlignedBuffer<std::remove_const_t<T>, N, Alignment>& buf) noexcept
        : data_(buf.data()), size_(N) {}
    
    // From AlignedVector
    AlignedSpan(AlignedVector<std::remove_const_t<T>>& vec) noexcept
        : data_(vec.data()), size_(vec.size()) {}
    
    AlignedSpan(const AlignedVector<std::remove_const_t<T>>& vec) noexcept
        : data_(vec.data()), size_(vec.size()) {}
    
    [[nodiscard]] pointer data() const noexcept { return data_; }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    
    [[nodiscard]] reference operator[](std::size_t i) const noexcept { return data_[i]; }
    
    [[nodiscard]] iterator begin() const noexcept { return data_; }
    [[nodiscard]] iterator end() const noexcept { return data_ + size_; }
    
    [[nodiscard]] AlignedSpan subspan(std::size_t offset, std::size_t count) const {
        // Note: subspan may not maintain alignment for arbitrary offsets
        // Only aligned if offset is multiple of (Alignment / sizeof(T))
        return AlignedSpan(data_ + offset, count);
    }
    
private:
    pointer data_;
    size_type size_;
};

// Deduction guides
template<typename T, std::size_t N, std::size_t A>
AlignedSpan(AlignedBuffer<T, N, A>&) -> AlignedSpan<T, A>;

template<typename T, std::size_t N, std::size_t A>
AlignedSpan(const AlignedBuffer<T, N, A>&) -> AlignedSpan<const T, A>;

} // namespace phantomcore
