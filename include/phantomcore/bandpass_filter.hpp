#pragma once

#include "types.hpp"
#include <array>
#include <cmath>
#include <numbers>

namespace phantomcore {

/**
 * @brief Second-Order Section (Biquad) IIR Filter
 * 
 * Implements Direct Form II Transposed for numerical stability.
 * Used as building block for higher-order Butterworth filters.
 */
class BiquadFilter {
public:
    // Coefficients: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
    float a1 = 0.0f, a2 = 0.0f;
    
    // State variables (Direct Form II Transposed)
    float z1 = 0.0f, z2 = 0.0f;
    
    /**
     * @brief Process single sample through biquad
     * @param x Input sample
     * @return Filtered output
     */
    [[nodiscard]] inline float process(float x) noexcept {
        float y = b0 * x + z1;
        z1 = b1 * x - a1 * y + z2;
        z2 = b2 * x - a2 * y;
        return y;
    }
    
    void reset() noexcept {
        z1 = z2 = 0.0f;
    }
};

/**
 * @brief 4th-order Butterworth Bandpass Filter
 * 
 * Designed for neural spike detection (300-3000Hz typical).
 * Implements cascade of 2 biquad sections for numerical stability.
 * 
 * Why Butterworth:
 * - Maximally flat passband (no ripple)
 * - Monotonic frequency response
 * - Good phase linearity in passband
 * - Standard in neuroscience signal processing
 */
class ButterworthBandpass {
public:
    struct Config {
        float sample_rate = 30000.0f;   // Hz (typical neural recording)
        float low_cutoff = 300.0f;      // Hz (removes LFP, motion artifacts)
        float high_cutoff = 3000.0f;    // Hz (removes high-freq noise)
        uint8_t order = 4;              // Filter order (must be 2 or 4)
    };
    
    explicit ButterworthBandpass(const Config& config = {}) 
        : config_(config) {
        design_filter();
    }
    
    /**
     * @brief Process single sample
     * @param x Raw input sample
     * @return Bandpass filtered output
     */
    [[nodiscard]] inline float process(float x) noexcept {
        // Cascade through all sections
        float y = x;
        for (auto& section : sections_) {
            y = section.process(y);
        }
        return y;
    }
    
    /**
     * @brief Process buffer in-place
     * @param buffer Input/output buffer
     * @param num_samples Number of samples
     */
    void process_buffer(float* buffer, size_t num_samples) noexcept {
        for (size_t i = 0; i < num_samples; ++i) {
            buffer[i] = process(buffer[i]);
        }
    }
    
    /**
     * @brief Reset filter state (call when starting new segment)
     */
    void reset() noexcept {
        for (auto& section : sections_) {
            section.reset();
        }
    }
    
    /**
     * @brief Reconfigure filter with new parameters
     */
    void reconfigure(const Config& config) {
        config_ = config;
        design_filter();
        reset();
    }
    
    const Config& config() const { return config_; }
    
private:
    Config config_;
    std::array<BiquadFilter, 4> sections_;  // Max 4 sections for 4th order BP
    size_t num_sections_ = 0;
    
    /**
     * @brief Design Butterworth bandpass filter coefficients
     * 
     * Uses bilinear transform with frequency pre-warping.
     * Bandpass is implemented as lowpass + highpass cascade.
     */
    void design_filter() {
        const float fs = config_.sample_rate;
        const float f1 = config_.low_cutoff;   // High-pass cutoff
        const float f2 = config_.high_cutoff;  // Low-pass cutoff
        
        // Pre-warp frequencies for bilinear transform
        const float w1 = std::tan(std::numbers::pi_v<float> * f1 / fs);
        const float w2 = std::tan(std::numbers::pi_v<float> * f2 / fs);
        
        if (config_.order == 2) {
            // 2nd order: 1 HPF biquad + 1 LPF biquad
            num_sections_ = 2;
            design_highpass_biquad(sections_[0], w1);
            design_lowpass_biquad(sections_[1], w2);
        } else {
            // 4th order: 2 HPF biquads + 2 LPF biquads (default)
            num_sections_ = 4;
            
            // Butterworth poles for 2nd order sections
            // For 4th order, we need 2 sections per filter type
            // Pole angles: π/8, 3π/8 for 4th order
            const float q1 = 1.0f / (2.0f * std::cos(std::numbers::pi_v<float> / 8.0f));
            const float q2 = 1.0f / (2.0f * std::cos(3.0f * std::numbers::pi_v<float> / 8.0f));
            
            design_highpass_biquad_q(sections_[0], w1, q1);
            design_highpass_biquad_q(sections_[1], w1, q2);
            design_lowpass_biquad_q(sections_[2], w2, q1);
            design_lowpass_biquad_q(sections_[3], w2, q2);
        }
    }
    
    /**
     * @brief Design 2nd-order Butterworth highpass section
     */
    static void design_highpass_biquad(BiquadFilter& bq, float wc) {
        // Q = 1/sqrt(2) for Butterworth
        design_highpass_biquad_q(bq, wc, std::numbers::sqrt2_v<float> / 2.0f);
    }
    
    /**
     * @brief Design 2nd-order highpass with specified Q
     */
    static void design_highpass_biquad_q(BiquadFilter& bq, float wc, float Q) {
        // Bilinear transform of s-domain highpass: H(s) = s^2 / (s^2 + s/Q + 1)
        const float wc2 = wc * wc;
        const float alpha = wc / Q;
        const float norm = 1.0f + alpha + wc2;
        
        bq.b0 = 1.0f / norm;
        bq.b1 = -2.0f / norm;
        bq.b2 = 1.0f / norm;
        bq.a1 = 2.0f * (wc2 - 1.0f) / norm;
        bq.a2 = (1.0f - alpha + wc2) / norm;
    }
    
    /**
     * @brief Design 2nd-order Butterworth lowpass section
     */
    static void design_lowpass_biquad(BiquadFilter& bq, float wc) {
        design_lowpass_biquad_q(bq, wc, std::numbers::sqrt2_v<float> / 2.0f);
    }
    
    /**
     * @brief Design 2nd-order lowpass with specified Q
     */
    static void design_lowpass_biquad_q(BiquadFilter& bq, float wc, float Q) {
        // Bilinear transform of s-domain lowpass: H(s) = 1 / (s^2 + s/Q + 1)
        const float wc2 = wc * wc;
        const float alpha = wc / Q;
        const float norm = 1.0f + alpha + wc2;
        
        bq.b0 = wc2 / norm;
        bq.b1 = 2.0f * wc2 / norm;
        bq.b2 = wc2 / norm;
        bq.a1 = 2.0f * (wc2 - 1.0f) / norm;
        bq.a2 = (1.0f - alpha + wc2) / norm;
    }
};

/**
 * @brief Multi-channel bandpass filter bank
 * 
 * Manages independent filter state per channel.
 * Required because each channel needs its own z1/z2 state.
 */
template<size_t NumChannels = NUM_CHANNELS>
class BandpassFilterBank {
public:
    using Config = ButterworthBandpass::Config;
    
    explicit BandpassFilterBank(const Config& config = {}) {
        for (auto& filter : filters_) {
            filter = ButterworthBandpass(config);
        }
    }
    
    /**
     * @brief Process single sample for one channel
     */
    [[nodiscard]] inline float process(size_t channel, float x) noexcept {
        return filters_[channel].process(x);
    }
    
    /**
     * @brief Process multi-channel sample in-place
     * @param sample Array of samples, one per channel
     */
    void process_multichannel(float* sample) noexcept {
        for (size_t ch = 0; ch < NumChannels; ++ch) {
            sample[ch] = filters_[ch].process(sample[ch]);
        }
    }
    
    /**
     * @brief Process multi-channel buffer
     * @param buffer Interleaved [sample0_ch0, sample0_ch1, ..., sample1_ch0, ...]
     * @param num_samples Number of multi-channel samples
     */
    void process_buffer_interleaved(float* buffer, size_t num_samples) noexcept {
        for (size_t s = 0; s < num_samples; ++s) {
            process_multichannel(buffer + s * NumChannels);
        }
    }
    
    /**
     * @brief Reset all filter states
     */
    void reset() noexcept {
        for (auto& filter : filters_) {
            filter.reset();
        }
    }
    
    /**
     * @brief Reconfigure all filters
     */
    void reconfigure(const Config& config) {
        for (auto& filter : filters_) {
            filter.reconfigure(config);
        }
    }
    
    ButterworthBandpass& channel_filter(size_t ch) { return filters_[ch]; }
    const ButterworthBandpass& channel_filter(size_t ch) const { return filters_[ch]; }
    
private:
    std::array<ButterworthBandpass, NumChannels> filters_;
};

} // namespace phantomcore
