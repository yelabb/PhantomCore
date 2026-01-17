#pragma once

#include "types.hpp"
#include <functional>
#include <memory>
#include <string>

namespace phantomcore {

/**
 * @brief WebSocket client for streaming neural data from PhantomLink
 * 
 * Features:
 * - Binary MessagePack protocol
 * - Automatic reconnection with exponential backoff
 * - Latency tracking
 * - Thread-safe packet handling
 */
class StreamClient {
public:
    /// Callback for received neural packets
    using PacketCallback = std::function<void(const NeuralPacket&)>;
    
    /// Callback for connection state changes
    using StateCallback = std::function<void(ConnectionState)>;
    
    /// Callback for errors
    using ErrorCallback = std::function<void(const std::string&)>;
    
    explicit StreamClient(const ConnectionConfig& config = {});
    ~StreamClient();
    
    // Non-copyable, movable
    StreamClient(const StreamClient&) = delete;
    StreamClient& operator=(const StreamClient&) = delete;
    StreamClient(StreamClient&&) noexcept;
    StreamClient& operator=(StreamClient&&) noexcept;
    
    /**
     * @brief Connect to PhantomLink server
     * @param session_code Optional session code (generates one if empty)
     * @return True if connection initiated successfully
     */
    bool connect(const std::string& session_code = "");
    
    /**
     * @brief Disconnect from server
     */
    void disconnect();
    
    /**
     * @brief Check if connected
     */
    bool is_connected() const;
    
    /**
     * @brief Get current connection state
     */
    ConnectionState state() const;
    
    /**
     * @brief Get current session code
     */
    std::string session_code() const;
    
    /**
     * @brief Set packet received callback
     * Called on the WebSocket thread - should be fast!
     */
    void on_packet(PacketCallback callback);
    
    /**
     * @brief Set connection state change callback
     */
    void on_state_change(StateCallback callback);
    
    /**
     * @brief Set error callback
     */
    void on_error(ErrorCallback callback);
    
    /**
     * @brief Get connection statistics
     */
    struct Stats {
        uint64_t packets_received = 0;
        uint64_t bytes_received = 0;
        uint64_t reconnect_count = 0;
        LatencyStats network_latency;
        Duration uptime{};
    };
    Stats get_stats() const;
    
    /**
     * @brief Send playback control command
     */
    bool send_pause();
    bool send_resume();
    bool send_seek(double timestamp);
    
    /**
     * @brief Update configuration
     */
    void set_config(const ConnectionConfig& config);
    const ConnectionConfig& config() const { return config_; }

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ConnectionConfig config_;
};

/**
 * @brief Generates a readable session code
 * Format: adjective-noun-number (e.g., "swift-neural-42")
 */
std::string generate_session_code();

/**
 * @brief Parses a MessagePack binary packet into NeuralPacket
 * @param data Raw binary data
 * @param size Data size in bytes
 * @return Parsed packet
 * @throws std::runtime_error on parse failure
 */
NeuralPacket parse_packet(const uint8_t* data, size_t size);

}  // namespace phantomcore
