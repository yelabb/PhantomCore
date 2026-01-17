/**
 * PhantomCore Spike Visualizer
 * 
 * Real-time console visualization of neural spike activity.
 * Displays activity heatmap across all 142 channels.
 */

#include <phantomcore.hpp>
#include <iostream>
#include <iomanip>
#include <thread>
#include <atomic>
#include <csignal>
#include <cmath>
#include <array>

using namespace phantomcore;

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

// ANSI color codes for heatmap
const char* get_color(int intensity) {
    // 0-9 intensity levels
    static const char* colors[] = {
        "\033[48;5;232m",  // 0 - dark
        "\033[48;5;234m",  // 1
        "\033[48;5;236m",  // 2
        "\033[48;5;238m",  // 3
        "\033[48;5;240m",  // 4
        "\033[48;5;22m",   // 5 - green start
        "\033[48;5;28m",   // 6
        "\033[48;5;34m",   // 7
        "\033[48;5;46m",   // 8 - bright green
        "\033[48;5;226m",  // 9 - yellow/high activity
    };
    return colors[std::min(9, std::max(0, intensity))];
}

const char* RESET = "\033[0m";

void clear_screen() {
#ifdef _WIN32
    system("cls");
#else
    std::cout << "\033[2J\033[H";
#endif
}

void move_cursor(int row, int col) {
    std::cout << "\033[" << row << ";" << col << "H";
}

class SpikeVisualizer {
public:
    void update(const SpikeCountArray& spikes) {
        for (size_t i = 0; i < NUM_CHANNELS; ++i) {
            // Exponential smoothing
            smoothed_[i] = 0.7f * smoothed_[i] + 0.3f * static_cast<float>(spikes[i]);
            
            // Track max for normalization
            if (smoothed_[i] > max_activity_) {
                max_activity_ = smoothed_[i];
            }
        }
        frame_count_++;
    }
    
    void render() {
        // Grid layout: 14 rows x 11 columns = 154 cells (142 channels + padding)
        const int ROWS = 14;
        const int COLS = 11;
        
        move_cursor(5, 1);
        
        std::cout << "┌";
        for (int c = 0; c < COLS; ++c) std::cout << "───";
        std::cout << "┐\n";
        
        for (int r = 0; r < ROWS; ++r) {
            std::cout << "│";
            for (int c = 0; c < COLS; ++c) {
                size_t ch = r * COLS + c;
                if (ch < NUM_CHANNELS) {
                    // Normalize to 0-9
                    float normalized = (max_activity_ > 0) 
                        ? smoothed_[ch] / max_activity_ * 9.0f 
                        : 0.0f;
                    int intensity = static_cast<int>(normalized);
                    
                    std::cout << get_color(intensity) << "   " << RESET;
                } else {
                    std::cout << "   ";
                }
            }
            std::cout << "│ ";
            
            // Row labels
            if (r == 0) std::cout << "High Activity";
            else if (r == 1) std::cout << get_color(9) << "   " << RESET << " 9+";
            else if (r == 2) std::cout << get_color(7) << "   " << RESET << " 5-8";
            else if (r == 3) std::cout << get_color(4) << "   " << RESET << " 2-4";
            else if (r == 4) std::cout << get_color(1) << "   " << RESET << " 0-1";
            else if (r == 5) std::cout << "Low Activity";
            else if (r == 7) std::cout << "Frame: " << frame_count_;
            else if (r == 8) std::cout << "Channels: " << NUM_CHANNELS;
            
            std::cout << "\n";
        }
        
        std::cout << "└";
        for (int c = 0; c < COLS; ++c) std::cout << "───";
        std::cout << "┘\n";
    }
    
private:
    std::array<float, NUM_CHANNELS> smoothed_{};
    float max_activity_ = 1.0f;
    uint64_t frame_count_ = 0;
};

int main(int argc, char* argv[]) {
    std::string server_url = "ws://localhost:8000/stream/binary/";
    if (argc > 1) server_url = argv[1];
    
    phantomcore::initialize();
    std::signal(SIGINT, signal_handler);
    
    clear_screen();
    
    std::cout << R"(
╔═══════════════════════════════════════════════════════════════╗
║              PhantomCore Neural Activity Monitor              ║
║                   142 Channel Heatmap View                    ║
╚═══════════════════════════════════════════════════════════════╝
)" << std::endl;

    ConnectionConfig config;
    config.url = server_url;
    
    StreamClient client(config);
    SpikeVisualizer visualizer;
    
    std::atomic<bool> new_data{false};
    SpikeCountArray latest_spikes{};
    
    client.on_packet([&](const NeuralPacket& packet) {
        latest_spikes = packet.spike_counts;
        new_data = true;
    });
    
    client.on_state_change([](ConnectionState state) {
        move_cursor(3, 1);
        const char* names[] = {"Disconnected", "Connecting", "Connected", "Reconnecting", "Error"};
        std::cout << "Status: " << names[static_cast<int>(state)] << "          ";
    });
    
    client.connect();
    
    // Render loop at ~20 FPS
    while (g_running) {
        if (new_data.exchange(false)) {
            visualizer.update(latest_spikes);
            visualizer.render();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    client.disconnect();
    phantomcore::shutdown();
    
    std::cout << "\n\nShutdown complete.\n";
    
    return 0;
}
