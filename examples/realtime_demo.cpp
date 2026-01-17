/**
 * PhantomCore Real-Time Demo
 * 
 * Connects to PhantomLink server and processes neural data in real-time.
 * Demonstrates sub-millisecond decode latency.
 */

#include <phantomcore.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <csignal>

using namespace phantomcore;

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

void print_banner() {
    std::cout << R"(
  ____  _                 _                   ____               
 |  _ \| |__   __ _ _ __ | |_ ___  _ __ ___  / ___|___  _ __ ___ 
 | |_) | '_ \ / _` | '_ \| __/ _ \| '_ ` _ \| |   / _ \| '__/ _ \
 |  __/| | | | (_| | | | | || (_) | | | | | | |__| (_) | | |  __/
 |_|   |_| |_|\__,_|_| |_|\__\___/|_| |_| |_|\____\___/|_|  \___|
                                                                  
    Ultra-Low-Latency Neural Signal Processing
    )" << std::endl;
}

void print_stats_header() {
    std::cout << "\n┌────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│  Packet  │ Decode (μs) │ Position (x,y)   │ Velocity (vx,vy)  │ Target │\n";
    std::cout << "├────────────────────────────────────────────────────────────────────────┤\n";
}

int main(int argc, char* argv[]) {
    print_banner();
    
    // Parse arguments
    std::string server_url = "ws://localhost:8000/stream/binary/";
    std::string session_code = "";
    
    if (argc > 1) server_url = argv[1];
    if (argc > 2) session_code = argv[2];
    
    // Initialize library
    phantomcore::initialize();
    std::cout << "SIMD: " << simd::simd_info() << "\n\n";
    
    // Set up signal handler
    std::signal(SIGINT, signal_handler);
    
    // Create components
    ConnectionConfig conn_config;
    conn_config.url = server_url;
    conn_config.auto_reconnect = true;
    
    StreamClient client(conn_config);
    KalmanDecoder decoder;
    SpikeDetector spike_detector;
    LatencyTracker total_latency_tracker;
    
    // Statistics
    std::atomic<uint64_t> packet_count{0};
    std::atomic<double> total_decode_us{0.0};
    
    // Set up callbacks
    client.on_state_change([](ConnectionState state) {
        const char* state_names[] = {
            "Disconnected", "Connecting", "Connected", "Reconnecting", "Error"
        };
        std::cout << "Connection: " << state_names[static_cast<int>(state)] << "\n";
    });
    
    client.on_error([](const std::string& error) {
        std::cerr << "Error: " << error << "\n";
    });
    
    client.on_packet([&](const NeuralPacket& packet) {
        auto start = Clock::now();
        
        // Process spikes
        auto spikes = spike_detector.process_spike_counts(
            packet.spike_counts,
            packet.timestamp
        );
        
        // Decode kinematics
        auto output = decoder.decode(packet.spike_counts);
        
        // Track total latency
        auto end = Clock::now();
        total_latency_tracker.record(end - start);
        
        double decode_us = to_microseconds(output.processing_time);
        total_decode_us += decode_us;
        uint64_t count = ++packet_count;
        
        // Print every 40th packet (1 per second at 40Hz)
        if (count % 40 == 0) {
            std::cout << "│ " << std::setw(8) << count
                      << " │ " << std::setw(11) << std::fixed << std::setprecision(1) << decode_us
                      << " │ (" << std::setw(6) << std::setprecision(2) << output.position.x
                      << ", " << std::setw(6) << output.position.y << ")"
                      << " │ (" << std::setw(6) << output.velocity.vx
                      << ", " << std::setw(6) << output.velocity.vy << ")"
                      << " │ " << std::setw(6) << packet.intention.target_id
                      << " │\n";
        }
    });
    
    // Connect
    std::cout << "Connecting to: " << server_url << "\n";
    if (!session_code.empty()) {
        std::cout << "Session: " << session_code << "\n";
    }
    
    if (!client.connect(session_code)) {
        std::cerr << "Failed to initiate connection\n";
        return 1;
    }
    
    print_stats_header();
    
    // Main loop
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Disconnect and print final stats
    client.disconnect();
    
    std::cout << "└────────────────────────────────────────────────────────────────────────┘\n\n";
    
    // Print statistics
    auto client_stats = client.get_stats();
    auto decoder_stats = decoder.get_stats();
    auto spike_stats = spike_detector.get_stats();
    auto latency_stats = total_latency_tracker.get_stats();
    
    std::cout << "=== Session Statistics ===\n\n";
    
    std::cout << "Network:\n";
    std::cout << "  Packets received: " << client_stats.packets_received << "\n";
    std::cout << "  Bytes received:   " << client_stats.bytes_received / 1024 << " KB\n";
    std::cout << "  Reconnects:       " << client_stats.reconnect_count << "\n\n";
    
    std::cout << "Decoder Performance:\n";
    std::cout << "  Total decodes:     " << decoder_stats.total_decodes << "\n";
    std::cout << "  Mean decode time:  " << to_microseconds(decoder_stats.mean_decode_time) << " μs\n";
    std::cout << "  Max decode time:   " << to_microseconds(decoder_stats.max_decode_time) << " μs\n\n";
    
    std::cout << "End-to-End Latency:\n";
    std::cout << "  Mean:  " << std::fixed << std::setprecision(2) << latency_stats.mean_us << " μs\n";
    std::cout << "  P50:   " << latency_stats.p50_us << " μs\n";
    std::cout << "  P95:   " << latency_stats.p95_us << " μs\n";
    std::cout << "  P99:   " << latency_stats.p99_us << " μs\n";
    std::cout << "  Max:   " << latency_stats.max_us << " μs\n\n";
    
    std::cout << "Spike Detection:\n";
    std::cout << "  Total spikes: " << spike_stats.total_spikes_detected << "\n";
    std::cout << "  Mean rate:    " << std::fixed << std::setprecision(1) 
              << spike_stats.mean_rate_hz << " Hz\n\n";
    
    phantomcore::shutdown();
    
    return 0;
}
