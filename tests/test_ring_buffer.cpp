#include <gtest/gtest.h>
#include <phantomcore/ring_buffer.hpp>
#include <phantomcore/types.hpp>  // For NeuralPacket
#include <thread>

using namespace phantomcore;

TEST(RingBufferTest, CreateEmpty) {
    RingBuffer<int, 16> buffer;
    EXPECT_TRUE(buffer.empty());
    EXPECT_FALSE(buffer.full());
    EXPECT_EQ(buffer.size(), 0u);
}

TEST(RingBufferTest, PushPop) {
    RingBuffer<int, 16> buffer;
    
    EXPECT_TRUE(buffer.push(42));
    EXPECT_EQ(buffer.size(), 1u);
    
    auto item = buffer.pop();
    EXPECT_TRUE(item.has_value());
    EXPECT_EQ(*item, 42);
    EXPECT_TRUE(buffer.empty());
}

TEST(RingBufferTest, FIFO) {
    RingBuffer<int, 16> buffer;
    
    for (int i = 0; i < 5; ++i) {
        buffer.push(i);
    }
    
    for (int i = 0; i < 5; ++i) {
        auto item = buffer.pop();
        EXPECT_EQ(*item, i);
    }
}

TEST(RingBufferTest, Full) {
    RingBuffer<int, 4> buffer;  // Capacity 4
    
    EXPECT_TRUE(buffer.push(1));
    EXPECT_TRUE(buffer.push(2));
    EXPECT_TRUE(buffer.push(3));
    EXPECT_TRUE(buffer.full());
    
    // Should fail when full
    EXPECT_FALSE(buffer.push(4));
}

TEST(RingBufferTest, PushOverwrite) {
    RingBuffer<int, 4> buffer;
    
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    // Now full, push with overwrite
    EXPECT_FALSE(buffer.push_overwrite(4));  // Returns false because it overwrote
    
    // First item should be gone
    auto item = buffer.pop();
    EXPECT_EQ(*item, 2);  // 1 was overwritten
}

TEST(RingBufferTest, Peek) {
    RingBuffer<int, 16> buffer;
    
    buffer.push(42);
    
    auto peeked = buffer.peek();
    EXPECT_TRUE(peeked.has_value());
    EXPECT_EQ(*peeked, 42);
    
    // Item should still be there
    EXPECT_EQ(buffer.size(), 1u);
}

TEST(RingBufferTest, Clear) {
    RingBuffer<int, 16> buffer;
    
    buffer.push(1);
    buffer.push(2);
    buffer.push(3);
    
    buffer.clear();
    
    EXPECT_TRUE(buffer.empty());
    EXPECT_EQ(buffer.size(), 0u);
}

TEST(RingBufferTest, Capacity) {
    RingBuffer<int, 64> buffer;
    EXPECT_EQ(buffer.capacity(), 64u);
}

TEST(RingBufferTest, WrapAround) {
    RingBuffer<int, 4> buffer;
    
    // Fill and empty multiple times to test wrap-around
    for (int round = 0; round < 10; ++round) {
        EXPECT_TRUE(buffer.push(round * 10 + 1));
        EXPECT_TRUE(buffer.push(round * 10 + 2));
        EXPECT_TRUE(buffer.push(round * 10 + 3));
        
        EXPECT_EQ(*buffer.pop(), round * 10 + 1);
        EXPECT_EQ(*buffer.pop(), round * 10 + 2);
        EXPECT_EQ(*buffer.pop(), round * 10 + 3);
    }
}

TEST(RingBufferTest, ThreadSafety) {
    RingBuffer<int, 1024> buffer;
    std::atomic<int> sum{0};
    
    const int N = 10000;
    
    // Producer thread
    std::thread producer([&]() {
        for (int i = 0; i < N; ++i) {
            while (!buffer.push(i)) {
                std::this_thread::yield();
            }
        }
    });
    
    // Consumer thread
    std::thread consumer([&]() {
        int count = 0;
        while (count < N) {
            auto item = buffer.pop();
            if (item.has_value()) {
                sum += *item;
                count++;
            } else {
                std::this_thread::yield();
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    // Sum should be 0 + 1 + 2 + ... + (N-1) = N*(N-1)/2
    int expected_sum = N * (N - 1) / 2;
    EXPECT_EQ(sum.load(), expected_sum);
}

TEST(RingBufferTest, NeuralPacketBuffer) {
    NeuralBuffer<NeuralPacket> buffer;
    
    NeuralPacket packet;
    packet.sequence = 42;
    packet.timestamp = 1.23;
    
    buffer.push(packet);
    
    auto retrieved = buffer.pop();
    EXPECT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->sequence, 42u);
    EXPECT_DOUBLE_EQ(retrieved->timestamp, 1.23);
}
