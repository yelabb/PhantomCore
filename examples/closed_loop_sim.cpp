/**
 * PhantomCore Closed-Loop Simulation
 * 
 * Demonstrates a complete closed-loop BCI system with:
 * - Neural signal acquisition (from PhantomLink)
 * - Real-time decoding (Kalman filter)
 * - Simulated feedback control
 * - Latency measurement at every stage
 */

#include <phantomcore.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <csignal>
#include <cmath>
#include <queue>
#include <mutex>

using namespace phantomcore;

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

// Simulated actuator/effector system
class SimulatedEffector {
public:
    struct Command {
        Vec2 target_position;
        Timestamp issued_at;
    };
    
    void send_command(const Vec2& position) {
        Command cmd;
        cmd.target_position = position;
        cmd.issued_at = Clock::now();
        
        std::lock_guard<std::mutex> lock(mutex_);
        command_queue_.push(cmd);
        total_commands_++;
    }
    
    void update() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (!command_queue_.empty()) {
            auto cmd = command_queue_.front();
            command_queue_.pop();
            
            // Simulate actuator dynamics (simple first-order)
            const float alpha = 0.1f;  // Response speed
            current_position_.x += alpha * (cmd.target_position.x - current_position_.x);
            current_position_.y += alpha * (cmd.target_position.y - current_position_.y);
            
            // Track latency
            auto latency = Clock::now() - cmd.issued_at;
            latency_tracker_.record(latency);
        }
    }
    
    Vec2 get_position() const { return current_position_; }
    LatencyStats get_latency_stats() const { return latency_tracker_.get_stats(); }
    uint64_t get_total_commands() const { return total_commands_; }
    
private:
    std::queue<Command> command_queue_;
    std::mutex mutex_;
    Vec2 current_position_;
    LatencyTracker latency_tracker_;
    std::atomic<uint64_t> total_commands_{0};
};

// Closed-loop controller
class ClosedLoopController {
public:
    ClosedLoopController(KalmanDecoder& decoder, SimulatedEffector& effector)
        : decoder_(decoder), effector_(effector) {}
    
    void process(const NeuralPacket& packet) {
        auto loop_start = Clock::now();
        
        // Stage 1: Decode
        auto decode_start = Clock::now();
        auto output = decoder_.decode(packet.spike_counts);
        decode_latency_.record(Clock::now() - decode_start);
        
        // Stage 2: Control computation (simple proportional control)
        auto control_start = Clock::now();
        Vec2 target = packet.intention.target_position;
        Vec2 current = output.position;
        
        // P controller: move towards target
        const float Kp = 0.5f;
        Vec2 error = {target.x - current.x, target.y - current.y};
        Vec2 command = {
            current.x + Kp * error.x,
            current.y + Kp * error.y
        };
        control_latency_.record(Clock::now() - control_start);
        
        // Stage 3: Send to effector
        effector_.send_command(command);
        
        // Track total loop latency
        total_loop_latency_.record(Clock::now() - loop_start);
        
        // Calculate tracking error
        float tracking_error = error.norm();
        total_error_ += tracking_error;
        loop_count_++;
    }
    
    void print_stats() const {
        auto decode_stats = decode_latency_.get_stats();
        auto control_stats = control_latency_.get_stats();
        auto loop_stats = total_loop_latency_.get_stats();
        auto effector_stats = effector_.get_latency_stats();
        
        std::cout << "\n=== Closed-Loop Performance ===\n\n";
        
        std::cout << "Stage Latencies (μs):\n";
        std::cout << "  ┌──────────────────┬──────────┬──────────┬──────────┐\n";
        std::cout << "  │ Stage            │   Mean   │   P95    │   P99    │\n";
        std::cout << "  ├──────────────────┼──────────┼──────────┼──────────┤\n";
        std::cout << "  │ Neural Decode    │" << std::setw(9) << std::fixed << std::setprecision(1) 
                  << decode_stats.mean_us << " │" << std::setw(9) << decode_stats.p95_us 
                  << " │" << std::setw(9) << decode_stats.p99_us << " │\n";
        std::cout << "  │ Control Compute  │" << std::setw(9) << control_stats.mean_us 
                  << " │" << std::setw(9) << control_stats.p95_us 
                  << " │" << std::setw(9) << control_stats.p99_us << " │\n";
        std::cout << "  │ Effector Queue   │" << std::setw(9) << effector_stats.mean_us 
                  << " │" << std::setw(9) << effector_stats.p95_us 
                  << " │" << std::setw(9) << effector_stats.p99_us << " │\n";
        std::cout << "  ├──────────────────┼──────────┼──────────┼──────────┤\n";
        std::cout << "  │ Total Loop       │" << std::setw(9) << loop_stats.mean_us 
                  << " │" << std::setw(9) << loop_stats.p95_us 
                  << " │" << std::setw(9) << loop_stats.p99_us << " │\n";
        std::cout << "  └──────────────────┴──────────┴──────────┴──────────┘\n\n";
        
        std::cout << "Control Performance:\n";
        std::cout << "  Loop iterations:  " << loop_count_ << "\n";
        std::cout << "  Mean track error: " << std::fixed << std::setprecision(3) 
                  << (loop_count_ > 0 ? total_error_ / loop_count_ : 0.0) << " units\n";
        std::cout << "  Effector commands:" << effector_.get_total_commands() << "\n\n";
        
        // Real-time assessment
        double total_latency_ms = loop_stats.p99_us / 1000.0;
        std::cout << "Real-Time Assessment:\n";
        if (total_latency_ms < 1.0) {
            std::cout << "  ✓ SUB-MILLISECOND closed-loop latency achieved!\n";
            std::cout << "  ✓ P99 total loop: " << std::fixed << std::setprecision(2) 
                      << total_latency_ms << " ms\n";
        } else if (total_latency_ms < 10.0) {
            std::cout << "  ✓ Real-time capable (< 10ms)\n";
        } else {
            std::cout << "  ⚠ Latency may impact real-time performance\n";
        }
    }
    
private:
    KalmanDecoder& decoder_;
    SimulatedEffector& effector_;
    
    LatencyTracker decode_latency_;
    LatencyTracker control_latency_;
    LatencyTracker total_loop_latency_;
    
    double total_error_ = 0.0;
    uint64_t loop_count_ = 0;
};

int main(int argc, char* argv[]) {
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    PhantomCore Closed-Loop Simulation                         ║
║                  Neural Decode → Control → Effector Pipeline                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
)" << std::endl;

    std::string server_url = "ws://localhost:8000/stream/binary/";
    if (argc > 1) server_url = argv[1];
    
    phantomcore::initialize();
    std::signal(SIGINT, signal_handler);
    
    std::cout << "SIMD: " << simd::simd_info() << "\n\n";
    
    // Create components
    KalmanDecoder decoder;
    SimulatedEffector effector;
    ClosedLoopController controller(decoder, effector);
    
    // Connection
    ConnectionConfig config;
    config.url = server_url;
    
    StreamClient client(config);
    
    std::atomic<uint64_t> packet_count{0};
    
    client.on_packet([&](const NeuralPacket& packet) {
        controller.process(packet);
        
        // Simulate effector update
        effector.update();
        
        uint64_t count = ++packet_count;
        
        // Progress indicator
        if (count % 40 == 0) {
            std::cout << "\rProcessed: " << count << " packets | "
                      << "Effector pos: (" 
                      << std::fixed << std::setprecision(2)
                      << effector.get_position().x << ", "
                      << effector.get_position().y << ")     " << std::flush;
        }
    });
    
    client.on_state_change([](ConnectionState state) {
        const char* names[] = {"Disconnected", "Connecting", "Connected", "Reconnecting", "Error"};
        std::cout << "Connection: " << names[static_cast<int>(state)] << "\n";
    });
    
    std::cout << "Connecting to: " << server_url << "\n";
    std::cout << "Press Ctrl+C to stop and see results.\n\n";
    
    client.connect();
    
    // Main loop
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    client.disconnect();
    
    std::cout << "\n";
    controller.print_stats();
    
    phantomcore::shutdown();
    
    return 0;
}
