#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <algorithm>
#include <cmath>

void cactus_int8_to_fp32(const int8_t* src, float* dst, size_t count, float scale) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE, 
        [src, dst, scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 16) * 16;
            float32x4_t scale_vec = vdupq_n_f32(scale);
            
            for (size_t i = start; i < simd_end; i += 16) {
                int8x16_t input = vld1q_s8(&src[i]);
                
                int16x8_t low = vmovl_s8(vget_low_s8(input));
                int16x8_t high = vmovl_s8(vget_high_s8(input));
                
                int32x4_t low_low = vmovl_s16(vget_low_s16(low));
                int32x4_t low_high = vmovl_s16(vget_high_s16(low));
                int32x4_t high_low = vmovl_s16(vget_low_s16(high));
                int32x4_t high_high = vmovl_s16(vget_high_s16(high));
                
                float32x4_t f_low_low = vcvtq_f32_s32(low_low);
                float32x4_t f_low_high = vcvtq_f32_s32(low_high);
                float32x4_t f_high_low = vcvtq_f32_s32(high_low);
                float32x4_t f_high_high = vcvtq_f32_s32(high_high);
                
                vst1q_f32(&dst[i], vmulq_f32(f_low_low, scale_vec));
                vst1q_f32(&dst[i + 4], vmulq_f32(f_low_high, scale_vec));
                vst1q_f32(&dst[i + 8], vmulq_f32(f_high_low, scale_vec));
                vst1q_f32(&dst[i + 12], vmulq_f32(f_high_high, scale_vec));
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<float>(src[i]) * scale;
            }
        });
}

void cactus_fp32_to_int8(const float* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, inv_scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 16) * 16;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_vec = vdupq_n_f32(-128.0f);
            float32x4_t max_vec = vdupq_n_f32(127.0f);
            
            for (size_t i = start; i < simd_end; i += 16) {
                float32x4_t input_0 = vld1q_f32(&src[i]);
                float32x4_t input_1 = vld1q_f32(&src[i + 4]);
                float32x4_t input_2 = vld1q_f32(&src[i + 8]);
                float32x4_t input_3 = vld1q_f32(&src[i + 12]);
                
                float32x4_t scaled_0 = vmulq_f32(input_0, inv_scale_vec);
                float32x4_t scaled_1 = vmulq_f32(input_1, inv_scale_vec);
                float32x4_t scaled_2 = vmulq_f32(input_2, inv_scale_vec);
                float32x4_t scaled_3 = vmulq_f32(input_3, inv_scale_vec);
                
                scaled_0 = vmaxq_f32(vminq_f32(scaled_0, max_vec), min_vec);
                scaled_1 = vmaxq_f32(vminq_f32(scaled_1, max_vec), min_vec);
                scaled_2 = vmaxq_f32(vminq_f32(scaled_2, max_vec), min_vec);
                scaled_3 = vmaxq_f32(vminq_f32(scaled_3, max_vec), min_vec);
                
                int32x4_t int_0 = vcvtnq_s32_f32(scaled_0); 
                int32x4_t int_1 = vcvtnq_s32_f32(scaled_1);
                int32x4_t int_2 = vcvtnq_s32_f32(scaled_2);
                int32x4_t int_3 = vcvtnq_s32_f32(scaled_3);
                
                int16x8_t int16_low = vcombine_s16(vqmovn_s32(int_0), vqmovn_s32(int_1));
                int16x8_t int16_high = vcombine_s16(vqmovn_s32(int_2), vqmovn_s32(int_3));
                
                int8x16_t result = vcombine_s8(vqmovn_s16(int16_low), vqmovn_s16(int16_high));
                vst1q_s8(&dst[i], result);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                float quantized = src[i] * inv_scale;
                dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
            }
        });
}

void cactus_dynamic_quantize_fp32_to_int8(const float* src, int8_t* dst, size_t count, float* computed_scale) {
    if (count == 0) return;
    
    float32x4_t abs_max_vec = vdupq_n_f32(0.0f);
    const size_t simd_end = (count / 4) * 4;
    
    for (size_t i = 0; i < simd_end; i += 4) {
        float32x4_t input = vld1q_f32(&src[i]);
        float32x4_t abs_input = vabsq_f32(input);
        abs_max_vec = vmaxq_f32(abs_max_vec, abs_input);
    }
    
    float abs_max = vmaxvq_f32(abs_max_vec);
    
    for (size_t i = simd_end; i < count; ++i) {
        abs_max = std::max(abs_max, std::abs(src[i]));
    }
    
    float scale = abs_max / 127.0f;
    if (scale == 0.0f) scale = 1.0f; 
    
    cactus_fp32_to_int8(src, dst, count, scale);
    
    if (computed_scale) *computed_scale = scale;
}

void cactus_fp16_to_fp32(const __fp16* src, float* dst, size_t count) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            
            for (size_t i = start; i < simd_end; i += 8) {
                float16x8_t input = vld1q_f16(&src[i]);
                
                float32x4_t output_low = vcvt_f32_f16(vget_low_f16(input));
                float32x4_t output_high = vcvt_f32_f16(vget_high_f16(input));
                
                vst1q_f32(&dst[i], output_low);
                vst1q_f32(&dst[i + 4], output_high);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<float>(src[i]);
            }
        });
}

void cactus_fp32_to_fp16(const float* src, __fp16* dst, size_t count) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            
            for (size_t i = start; i < simd_end; i += 8) {
                float32x4_t input_low = vld1q_f32(&src[i]);
                float32x4_t input_high = vld1q_f32(&src[i + 4]);
                
                float16x4_t output_low = vcvt_f16_f32(input_low);
                float16x4_t output_high = vcvt_f16_f32(input_high);
                
                float16x8_t output = vcombine_f16(output_low, output_high);
                vst1q_f16(&dst[i], output);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<__fp16>(src[i]);
            }
        });
}

void cactus_int8_to_fp16(const int8_t* src, __fp16* dst, size_t count, float scale) {
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            float32x4_t scale_vec = vdupq_n_f32(scale);
            
            for (size_t i = start; i < simd_end; i += 8) {
                int8x8_t input = vld1_s8(&src[i]);
                
                int16x8_t int16 = vmovl_s8(input);
                int32x4_t int32_low = vmovl_s16(vget_low_s16(int16));
                int32x4_t int32_high = vmovl_s16(vget_high_s16(int16));
                
                float32x4_t float_low = vcvtq_f32_s32(int32_low);
                float32x4_t float_high = vcvtq_f32_s32(int32_high);
                
                float_low = vmulq_f32(float_low, scale_vec);
                float_high = vmulq_f32(float_high, scale_vec);
                
                float16x8_t output = vcombine_f16(vcvt_f16_f32(float_low), vcvt_f16_f32(float_high));
                vst1q_f16(&dst[i], output);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                dst[i] = static_cast<__fp16>(static_cast<float>(src[i]) * scale);
            }
        });
}

void cactus_fp16_to_int8(const __fp16* src, int8_t* dst, size_t count, float scale) {
    const float inv_scale = 1.0f / scale;
    
    CactusThreading::parallel_for(count, CactusThreading::Thresholds::ELEMENT_WISE,
        [src, dst, inv_scale](size_t start, size_t end) {
            const size_t simd_end = start + ((end - start) / 8) * 8;
            float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);
            float32x4_t min_vec = vdupq_n_f32(-128.0f);
            float32x4_t max_vec = vdupq_n_f32(127.0f);
            
            for (size_t i = start; i < simd_end; i += 8) {
                float16x8_t input = vld1q_f16(&src[i]);
                
                float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input));
                float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input));
                
                float32x4_t scaled_low = vmulq_f32(input_low, inv_scale_vec);
                float32x4_t scaled_high = vmulq_f32(input_high, inv_scale_vec);
                
                scaled_low = vmaxq_f32(vminq_f32(scaled_low, max_vec), min_vec);
                scaled_high = vmaxq_f32(vminq_f32(scaled_high, max_vec), min_vec);
                
                int32x4_t int_low = vcvtnq_s32_f32(scaled_low);
                int32x4_t int_high = vcvtnq_s32_f32(scaled_high);
                
                int16x8_t int16_combined = vcombine_s16(vqmovn_s32(int_low), vqmovn_s32(int_high));
                int8x8_t result = vqmovn_s16(int16_combined);
                
                vst1_s8(&dst[i], result);
            }
            
            for (size_t i = simd_end; i < end; ++i) {
                float quantized = static_cast<float>(src[i]) * inv_scale;
                dst[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, quantized))));
            }
        });
}

float cactus_fp16_max_abs(const __fp16* src, size_t count) {
    float32x4_t abs_max_vec = vdupq_n_f32(0.0f);
    const size_t simd_end = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_end; i += 8) {
        float16x8_t input = vld1q_f16(&src[i]);
        
        float32x4_t input_low = vcvt_f32_f16(vget_low_f16(input));
        float32x4_t input_high = vcvt_f32_f16(vget_high_f16(input));
        
        float32x4_t abs_low = vabsq_f32(input_low);
        float32x4_t abs_high = vabsq_f32(input_high);
        
        abs_max_vec = vmaxq_f32(abs_max_vec, abs_low);
        abs_max_vec = vmaxq_f32(abs_max_vec, abs_high);
    }
    
    float max_abs = vmaxvq_f32(abs_max_vec);
    
    for (size_t i = simd_end; i < count; ++i) {
        float abs_val = std::abs(static_cast<float>(src[i]));
        max_abs = std::max(max_abs, abs_val);
    }
    
    return max_abs;
}

void cactus_int32_to_fp16_scaled(const int32_t* src, __fp16* dst, size_t count, float scale) {
    float32x4_t scale_vec = vdupq_n_f32(scale);
    const size_t simd_end = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_end; i += 8) {
        int32x4_t int_low = vld1q_s32(&src[i]);
        int32x4_t int_high = vld1q_s32(&src[i + 4]);
        
        float32x4_t fp32_low = vcvtq_f32_s32(int_low);
        float32x4_t fp32_high = vcvtq_f32_s32(int_high);
        
        float32x4_t scaled_low = vmulq_f32(fp32_low, scale_vec);
        float32x4_t scaled_high = vmulq_f32(fp32_high, scale_vec);
        
        float16x8_t result = vcombine_f16(vcvt_f16_f32(scaled_low), vcvt_f16_f32(scaled_high));
        vst1q_f16(&dst[i], result);
    }
    
    for (size_t i = simd_end; i < count; ++i) {
        float fp32_val = static_cast<float>(src[i]) * scale;
        dst[i] = static_cast<__fp16>(fp32_val);
    }
}

// INT4 unpacking: converts packed INT4 (2 values per byte) to INT8
// Pack format: low nibble = even index, high nibble = odd index
// Signed INT4 range: -8 to +7 (stored as 0-15, where 8-15 = -8 to -1)
void cactus_unpack_int4_to_int8(const uint8_t* packed, int8_t* unpacked, size_t num_elements) {
    size_t i = 0;

    // NEON path: process 32 INT4 values (16 packed bytes) at a time
    for (; i + 32 <= num_elements; i += 32) {
        // Load 16 packed bytes containing 32 INT4 values
        uint8x16_t bytes = vld1q_u8(&packed[i / 2]);

        // Extract low nibbles (even indices): value & 0x0F
        uint8x16_t low = vandq_u8(bytes, vdupq_n_u8(0x0F));

        // Extract high nibbles (odd indices): value >> 4
        uint8x16_t high = vshrq_n_u8(bytes, 4);

        // Sign extend: if nibble >= 8, subtract 16 to get signed value
        // Reinterpret as signed for comparison
        int8x16_t low_s = vreinterpretq_s8_u8(low);
        int8x16_t high_s = vreinterpretq_s8_u8(high);

        int8x16_t eight = vdupq_n_s8(8);
        int8x16_t sixteen = vdupq_n_s8(16);

        // Create masks for values >= 8
        uint8x16_t low_needs_adjust = vcgeq_s8(low_s, eight);
        uint8x16_t high_needs_adjust = vcgeq_s8(high_s, eight);

        // Subtract 16 where needed for sign extension
        low_s = vsubq_s8(low_s, vandq_s8(sixteen, vreinterpretq_s8_u8(low_needs_adjust)));
        high_s = vsubq_s8(high_s, vandq_s8(sixteen, vreinterpretq_s8_u8(high_needs_adjust)));

        // Interleave: result[2i] = low, result[2i+1] = high
        int8x16x2_t interleaved = vzipq_s8(low_s, high_s);

        // Store 32 INT8 values
        vst1q_s8(&unpacked[i], interleaved.val[0]);
        vst1q_s8(&unpacked[i + 16], interleaved.val[1]);
    }

    // Scalar fallback for remaining elements (process pairs)
    for (; i + 2 <= num_elements; i += 2) {
        uint8_t byte = packed[i / 2];

        // Low nibble (even index)
        int8_t low = byte & 0x0F;
        if (low >= 8) low -= 16;

        // High nibble (odd index)
        int8_t high = (byte >> 4) & 0x0F;
        if (high >= 8) high -= 16;

        unpacked[i] = low;
        unpacked[i + 1] = high;
    }

    // Handle odd element count (last element in low nibble only)
    if (i < num_elements) {
        uint8_t byte = packed[i / 2];
        int8_t low = byte & 0x0F;
        if (low >= 8) low -= 16;
        unpacked[i] = low;
    }
}