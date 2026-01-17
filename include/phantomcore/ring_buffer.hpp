#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>

namespace phantomcore {

/**
 * @brief Lock-free single-producer single-consumer ring buffer
 * 
 * Optimized for streaming applications where one thread produces
 * data and another consumes it. Uses atomic operations for
 * thread-safety without locks.
 * 
 * @tparam T Element type (should be trivially copyable)
 * @tparam Capacity Buffer capacity (must be power of 2)
 */
template<typename T, size_t Capacity>
class RingBuffer {
    static_assert((Capacity & (Capacity - 1)) == 0, 
                  "Capacity must be a power of 2");
    
public:
    RingBuffer() = default;
    
    /**
     * @brief Push element to buffer
     * @param item Element to push
     * @return True if successful, false if buffer is full
     */
    bool push(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        
        if (next_head == tail_.load(std::memory_order_acquire)) {
            return false;  // Buffer full
        }
        
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return true;
    }
    
    /**
     * @brief Push element, overwriting oldest if full
     * @param item Element to push
     * @return True if pushed without overwrite
     */
    bool push_overwrite(const T& item) {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        
        bool overwrote = false;
        if (next_head == tail_.load(std::memory_order_acquire)) {
            // Advance tail to make room
            tail_.store((tail_.load(std::memory_order_relaxed) + 1) & (Capacity - 1),
                       std::memory_order_release);
            overwrote = true;
        }
        
        buffer_[head] = item;
        head_.store(next_head, std::memory_order_release);
        return !overwrote;
    }
    
    /**
     * @brief Pop element from buffer
     * @return Element if available, nullopt if empty
     */
    std::optional<T> pop() {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;  // Buffer empty
        }
        
        T item = buffer_[tail];
        tail_.store((tail + 1) & (Capacity - 1), std::memory_order_release);
        return item;
    }
    
    /**
     * @brief Peek at front element without removing
     * @return Element if available, nullopt if empty
     */
    std::optional<T> peek() const {
        const size_t tail = tail_.load(std::memory_order_relaxed);
        
        if (tail == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }
        
        return buffer_[tail];
    }
    
    /**
     * @brief Check if buffer is empty
     */
    bool empty() const {
        return head_.load(std::memory_order_acquire) == 
               tail_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Check if buffer is full
     */
    bool full() const {
        const size_t head = head_.load(std::memory_order_relaxed);
        const size_t next_head = (head + 1) & (Capacity - 1);
        return next_head == tail_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get number of elements in buffer
     */
    size_t size() const {
        const size_t head = head_.load(std::memory_order_acquire);
        const size_t tail = tail_.load(std::memory_order_acquire);
        return (head - tail + Capacity) & (Capacity - 1);
    }
    
    /**
     * @brief Get buffer capacity
     */
    static constexpr size_t capacity() { return Capacity; }
    
    /**
     * @brief Clear all elements
     */
    void clear() {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    std::array<T, Capacity> buffer_;
    alignas(64) std::atomic<size_t> head_{0};  // Cache line aligned
    alignas(64) std::atomic<size_t> tail_{0};  // Separate cache line
};

/**
 * @brief Ring buffer optimized for neural packets
 * 
 * Pre-sized for 1 second of data at 40Hz with some headroom
 */
template<typename T>
using NeuralBuffer = RingBuffer<T, 64>;

/**
 * @brief Ring buffer for latency samples
 */
template<typename T>
using LatencyBuffer = RingBuffer<T, 1024>;

}  // namespace phantomcore
