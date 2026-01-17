#include "phantomcore/stream_client.hpp"
#include "phantomcore/latency_tracker.hpp"
#include <ixwebsocket/IXWebSocket.h>
#include <msgpack.hpp>
#include <atomic>
#include <mutex>
#include <random>
#include <sstream>
#include <thread>

namespace phantomcore {

// Word lists for session code generation
static const char* ADJECTIVES[] = {
    "swift", "neural", "cosmic", "quantum", "phantom",
    "cyber", "alpha", "omega", "delta", "sigma",
    "rapid", "bright", "deep", "prime", "ultra"
};

static const char* NOUNS[] = {
    "cortex", "synapse", "neuron", "pulse", "wave",
    "signal", "link", "stream", "flow", "spark",
    "matrix", "nexus", "core", "hub", "net"
};

std::string generate_session_code() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<size_t> adj_dist(0, 14);
    std::uniform_int_distribution<size_t> noun_dist(0, 14);
    std::uniform_int_distribution<int> num_dist(10, 99);
    
    std::ostringstream ss;
    ss << ADJECTIVES[adj_dist(gen)] << "-"
       << NOUNS[noun_dist(gen)] << "-"
       << num_dist(gen);
    
    return ss.str();
}

// ============================================================================
// MessagePack Packet Parser
// ============================================================================

NeuralPacket parse_packet(const uint8_t* data, size_t size) {
    NeuralPacket packet;
    packet.received_at = Clock::now();
    
    try {
        msgpack::object_handle oh = msgpack::unpack(
            reinterpret_cast<const char*>(data), size
        );
        msgpack::object obj = oh.get();
        
        if (obj.type != msgpack::type::MAP) {
            throw std::runtime_error("Expected MAP at root");
        }
        
        auto root = obj.as<std::map<std::string, msgpack::object>>();
        
        // Parse metadata
        if (root.count("metadata")) {
            auto meta = root["metadata"].as<std::map<std::string, msgpack::object>>();
            if (meta.count("sequence")) packet.sequence = meta["sequence"].as<uint64_t>();
            if (meta.count("trial_id")) packet.trial_id = meta["trial_id"].as<uint64_t>();
            if (meta.count("timestamp")) packet.timestamp = meta["timestamp"].as<double>();
        }
        
        // Parse data
        if (root.count("data")) {
            auto data_obj = root["data"].as<std::map<std::string, msgpack::object>>();
            
            // Spikes
            if (data_obj.count("spikes")) {
                auto spikes = data_obj["spikes"].as<std::map<std::string, msgpack::object>>();
                if (spikes.count("spike_counts")) {
                    auto counts = spikes["spike_counts"].as<std::vector<int32_t>>();
                    for (size_t i = 0; i < std::min(counts.size(), NUM_CHANNELS); ++i) {
                        packet.spike_counts[i] = counts[i];
                    }
                }
            }
            
            // Kinematics
            if (data_obj.count("kinematics")) {
                auto kin = data_obj["kinematics"].as<std::map<std::string, msgpack::object>>();
                if (kin.count("x")) packet.kinematics.position.x = kin["x"].as<float>();
                if (kin.count("y")) packet.kinematics.position.y = kin["y"].as<float>();
                if (kin.count("vx")) packet.kinematics.velocity.vx = kin["vx"].as<float>();
                if (kin.count("vy")) packet.kinematics.velocity.vy = kin["vy"].as<float>();
            }
            
            // Intention
            if (data_obj.count("intention")) {
                auto intent = data_obj["intention"].as<std::map<std::string, msgpack::object>>();
                if (intent.count("target_id")) packet.intention.target_id = intent["target_id"].as<int32_t>();
                if (intent.count("target_x")) packet.intention.target_position.x = intent["target_x"].as<float>();
                if (intent.count("target_y")) packet.intention.target_position.y = intent["target_y"].as<float>();
                if (intent.count("distance")) packet.intention.distance = intent["distance"].as<float>();
            }
        }
        
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Packet parse error: ") + e.what());
    }
    
    return packet;
}

// ============================================================================
// StreamClient Implementation
// ============================================================================

struct StreamClient::Impl {
    ix::WebSocket websocket;
    
    std::atomic<ConnectionState> state{ConnectionState::Disconnected};
    std::string session_code;
    
    PacketCallback on_packet_cb;
    StateCallback on_state_cb;
    ErrorCallback on_error_cb;
    
    std::mutex callback_mutex;
    
    // Statistics
    Stats stats{};
    LatencyTracker latency_tracker{1000};
    Timestamp connect_time;
    
    // Reconnection
    uint32_t reconnect_attempts = 0;
    
    void handle_message(const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Message) {
            if (msg->binary) {
                try {
                    NeuralPacket packet = parse_packet(
                        reinterpret_cast<const uint8_t*>(msg->str.data()),
                        msg->str.size()
                    );
                    
                    stats.packets_received++;
                    stats.bytes_received += msg->str.size();
                    
                    // Track network latency (if server timestamp available)
                    latency_tracker.record(Clock::now() - packet.received_at);
                    
                    std::lock_guard<std::mutex> lock(callback_mutex);
                    if (on_packet_cb) {
                        on_packet_cb(packet);
                    }
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(callback_mutex);
                    if (on_error_cb) {
                        on_error_cb(std::string("Parse error: ") + e.what());
                    }
                }
            }
        }
    }
    
    void set_state(ConnectionState new_state) {
        state.store(new_state);
        std::lock_guard<std::mutex> lock(callback_mutex);
        if (on_state_cb) {
            on_state_cb(new_state);
        }
    }
};

StreamClient::StreamClient(const ConnectionConfig& config)
    : impl_(std::make_unique<Impl>()), config_(config) {
    
    impl_->websocket.setOnMessageCallback(
        [this](const ix::WebSocketMessagePtr& msg) {
            switch (msg->type) {
                case ix::WebSocketMessageType::Open:
                    impl_->connect_time = Clock::now();
                    impl_->set_state(ConnectionState::Connected);
                    impl_->reconnect_attempts = 0;
                    break;
                    
                case ix::WebSocketMessageType::Close:
                    if (config_.auto_reconnect && 
                        impl_->reconnect_attempts < config_.max_reconnect_attempts) {
                        impl_->set_state(ConnectionState::Reconnecting);
                        impl_->reconnect_attempts++;
                    } else {
                        impl_->set_state(ConnectionState::Disconnected);
                    }
                    break;
                    
                case ix::WebSocketMessageType::Error:
                    {
                        std::lock_guard<std::mutex> lock(impl_->callback_mutex);
                        if (impl_->on_error_cb) {
                            impl_->on_error_cb(msg->errorInfo.reason);
                        }
                    }
                    impl_->set_state(ConnectionState::Error);
                    break;
                    
                case ix::WebSocketMessageType::Message:
                    impl_->handle_message(msg);
                    break;
                    
                default:
                    break;
            }
        }
    );
}

StreamClient::~StreamClient() {
    disconnect();
}

StreamClient::StreamClient(StreamClient&&) noexcept = default;
StreamClient& StreamClient::operator=(StreamClient&&) noexcept = default;

bool StreamClient::connect(const std::string& session_code) {
    if (is_connected()) {
        return true;
    }
    
    impl_->session_code = session_code.empty() ? generate_session_code() : session_code;
    
    std::string url = config_.url + impl_->session_code;
    impl_->websocket.setUrl(url);
    
    // Enable binary messages
    impl_->websocket.disableAutomaticReconnection();
    
    impl_->set_state(ConnectionState::Connecting);
    impl_->websocket.start();
    
    return true;
}

void StreamClient::disconnect() {
    impl_->websocket.stop();
    impl_->set_state(ConnectionState::Disconnected);
}

bool StreamClient::is_connected() const {
    return impl_->state.load() == ConnectionState::Connected;
}

ConnectionState StreamClient::state() const {
    return impl_->state.load();
}

std::string StreamClient::session_code() const {
    return impl_->session_code;
}

void StreamClient::on_packet(PacketCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->on_packet_cb = std::move(callback);
}

void StreamClient::on_state_change(StateCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->on_state_cb = std::move(callback);
}

void StreamClient::on_error(ErrorCallback callback) {
    std::lock_guard<std::mutex> lock(impl_->callback_mutex);
    impl_->on_error_cb = std::move(callback);
}

StreamClient::Stats StreamClient::get_stats() const {
    Stats stats = impl_->stats;
    stats.network_latency = impl_->latency_tracker.get_stats();
    stats.reconnect_count = impl_->reconnect_attempts;
    
    if (is_connected()) {
        stats.uptime = Clock::now() - impl_->connect_time;
    }
    
    return stats;
}

bool StreamClient::send_pause() {
    if (!is_connected()) return false;
    impl_->websocket.send(R"({"command": "pause"})");
    return true;
}

bool StreamClient::send_resume() {
    if (!is_connected()) return false;
    impl_->websocket.send(R"({"command": "resume"})");
    return true;
}

bool StreamClient::send_seek(double timestamp) {
    if (!is_connected()) return false;
    std::ostringstream ss;
    ss << R"({"command": "seek", "timestamp": )" << timestamp << "}";
    impl_->websocket.send(ss.str());
    return true;
}

void StreamClient::set_config(const ConnectionConfig& config) {
    config_ = config;
}

}  // namespace phantomcore
