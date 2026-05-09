// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// ============================================================================
// GGUF Dequantization - Intel XPU ESIMD Optimized Implementation
// ============================================================================
// Ported from llm-scaler (omni_xpu_kernel/csrc/gguf_dequant.cpp).
// Stripped of PyTorch surface; kernel bodies kept identical for side-by-side
// comparison against the source. Variable and constant names preserved.
//
// Supported formats (matching ComfyUI-GGUF / GGML layout):
//   Q4_0: Block=32, Size=18 bytes (2 scale + 16 data)
//   Q4_K: Block=256, Size=144 bytes (2+2+12+128)
// ============================================================================

#ifndef GGML_SYCL_DEQUANTIZE_ESIMD_HPP
#define GGML_SYCL_DEQUANTIZE_ESIMD_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "common.hpp"

namespace omni_xpu {
namespace gguf {

using fp16 = sycl::half;
#ifdef GGML_SYCL_HAS_BF16
using bf16 = sycl::ext::oneapi::bfloat16;
#endif
using namespace sycl::ext::intel::esimd;

// Q4_0: 18 bytes per block, 32 elements
constexpr int Q4_0_BLOCK_SIZE = 18;
constexpr int Q4_0_QK = 32;

// Q4_K: 144 bytes per block, 256 elements
// (K_SCALE_SIZE comes from ggml-common.h, value 12)
constexpr int Q4_K_BLOCK_SIZE = 144;
constexpr int Q4_K_QK = 256;

// ============================================================================
// Q4_0 Kernel (ComfyUI Sequential Layout)
// Output layout: [0-15]=low nibbles, [16-31]=high nibbles
// ============================================================================
template<typename OT, int SBS = 16>
void dequantize_q4_0_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    sycl::queue * stream
) {
    constexpr int WG_SIZE = 64;
    const int64_t n_work_items = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_size = (n_work_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_work_items) return;

                const int64_t start_block = gid * SBS;
                const int64_t end_block = std::min(start_block + SBS, n_blocks);

                simd<uint32_t, 16> offsets;
                #pragma unroll
                for (int i = 0; i < 16; ++i) offsets[i] = i;

                for (int64_t blk = start_block; blk < end_block; ++blk) {
                    const uint8_t* block_src = src + blk * Q4_0_BLOCK_SIZE;
                    OT* block_dst = dst + blk * Q4_0_QK;

                    const fp16 scale = *reinterpret_cast<const fp16*>(block_src);
                    simd<uint8_t, 16> packed = gather<uint8_t, 16>(block_src + 2, offsets);

                    // Sequential layout: low nibbles first, then high nibbles
                    simd<uint8_t, Q4_0_QK> unpacked;
                    unpacked.select<16, 1>(0) = packed & (uint8_t)0x0F;
                    unpacked.select<16, 1>(16) = packed >> 4;

                    simd<int16_t, Q4_0_QK> signed_vals = unpacked;
                    signed_vals = signed_vals - (int16_t)8;
                    simd<fp16, Q4_0_QK> result = signed_vals * scale;

                    if constexpr (std::is_same_v<OT, fp16>) {
                        block_store<fp16, Q4_0_QK>(reinterpret_cast<fp16*>(block_dst), result);
                    }
#ifdef GGML_SYCL_HAS_BF16
                    else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, Q4_0_QK> bf_result = result;
                        block_store<bf16, Q4_0_QK>(reinterpret_cast<bf16*>(block_dst), bf_result);
                    }
#endif
                    else {
                        simd<float, Q4_0_QK> f_result = result;
                        block_store<float, Q4_0_QK>(block_dst, f_result);
                    }
                }
            }
        );
    };

    stream->submit(cgf);
}

// ============================================================================
// Q4_K Kernel (ComfyUI layout)
//
// Deviation from source: intermediate scale/min math runs in float (fp32)
// to match GGML's reference precision (ggml-quants.c dequantize_row_q4_K
// and ggml-sycl's dequantize_block_q4_K, both of which keep dall/dmin in
// fp32). The source kernel does this in fp16 which drifts perplexity by a
// small but measurable amount on Q4_K_M MoE.
// ============================================================================
template<typename OT>
void dequantize_q4_k_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    sycl::queue * stream
) {
    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_blocks + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t blk = item.get_global_id(0);
                if (blk >= n_blocks) return;

                const uint8_t* block_src = src + blk * Q4_K_BLOCK_SIZE;
                OT* block_dst = dst + blk * Q4_K_QK;

                const float d    = float(*reinterpret_cast<const fp16*>(block_src));
                const float dmin = float(*reinterpret_cast<const fp16*>(block_src + 2));

                simd<uint8_t, K_SCALE_SIZE> scales_data;
                const uint8_t* scales_ptr = block_src + 4;
                #pragma unroll
                for (int i = 0; i < K_SCALE_SIZE; ++i) scales_data[i] = scales_ptr[i];

                const uint8_t* qs = block_src + 4 + K_SCALE_SIZE;

                simd<uint32_t, 32> offsets32;
                #pragma unroll
                for (int i = 0; i < 32; ++i) offsets32[i] = i;

                // Process 4 super-groups (PyTorch layout)
                #pragma unroll
                for (int sg = 0; sg < 4; ++sg) {
                    const int j_low = sg * 2;
                    const int j_high = sg * 2 + 1;

                    // Extract scales and mins
                    uint8_t sc_low, m_low, sc_high, m_high;
                    if (j_low < 4) {
                        sc_low = scales_data[j_low] & 63;
                        m_low = scales_data[j_low + 4] & 63;
                    } else {
                        sc_low = (scales_data[j_low + 4] & 0xF) | ((scales_data[j_low - 4] >> 2) & 0x30);
                        m_low = (scales_data[j_low + 4] >> 4) | ((scales_data[j_low] >> 2) & 0x30);
                    }
                    if (j_high < 4) {
                        sc_high = scales_data[j_high] & 63;
                        m_high = scales_data[j_high + 4] & 63;
                    } else {
                        sc_high = (scales_data[j_high + 4] & 0xF) | ((scales_data[j_high - 4] >> 2) & 0x30);
                        m_high = (scales_data[j_high + 4] >> 4) | ((scales_data[j_high] >> 2) & 0x30);
                    }

                    float d_sc_low  = d    * float(sc_low);
                    float dm_m_low  = dmin * float(m_low);
                    float d_sc_high = d    * float(sc_high);
                    float dm_m_high = dmin * float(m_high);

                    simd<uint8_t, 32> packed = gather<uint8_t, 32>(qs + sg * 32, offsets32);
                    simd<uint8_t, 32> low_nibbles = packed & (uint8_t)0x0F;
                    simd<uint8_t, 32> high_nibbles = packed >> 4;

                    simd<float, 32> result_low, result_high;
                    #pragma unroll
                    for (int i = 0; i < 32; ++i) {
                        result_low[i]  = d_sc_low  * float(low_nibbles[i])  - dm_m_low;
                        result_high[i] = d_sc_high * float(high_nibbles[i]) - dm_m_high;
                    }

                    OT* out_low = block_dst + j_low * 32;
                    OT* out_high = block_dst + j_high * 32;

                    if constexpr (std::is_same_v<OT, fp16>) {
                        simd<fp16, 32> hf_low = result_low, hf_high = result_high;
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_low), hf_low);
                        block_store<fp16, 32>(reinterpret_cast<fp16*>(out_high), hf_high);
                    }
#ifdef GGML_SYCL_HAS_BF16
                    else if constexpr (std::is_same_v<OT, bf16>) {
                        simd<bf16, 32> bf_low = result_low, bf_high = result_high;
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_low), bf_low);
                        block_store<bf16, 32>(reinterpret_cast<bf16*>(out_high), bf_high);
                    }
#endif
                    else {
                        block_store<float, 32>(out_low, result_low);
                        block_store<float, 32>(out_high, result_high);
                    }
                }
            }
        );
    };

    stream->submit(cgf);
}

// ============================================================================
// Q4_K Reorder Kernel (ggml-sycl reorder layout)
//
// Layout differs from the standard Q4_K_M packing: instead of 144-byte blocks
// of [d|dmin|scales|qs], the three regions are split into contiguous arrays
// across all blocks:
//     [qs section: n_blocks * 128 bytes]
//     [scales section: n_blocks * 12 bytes]
//     [dm section: n_blocks * 4 bytes (fp16 d, fp16 dmin)]
//
// Per-block compute is identical to dequantize_q4_k_kernel; only the input
// pointer arithmetic differs. Output keeps fp32 intermediates to match the
// GGML reference (see comment on dequantize_q4_k_kernel).
//
// No source kernel exists upstream for this layout — original ESIMD work for
// ggml-sycl's reorder optimization.
// ============================================================================
template<typename OT, int SBS = 4>
void dequantize_q4_k_reorder_kernel(
    const uint8_t* __restrict__ src,
    OT* __restrict__ dst,
    const int64_t n_blocks,
    sycl::queue * stream
) {
    constexpr int WG_SIZE = 64;
    const int64_t n_work_items = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_size = (n_work_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    const uint8_t * qs_base     = src;
    const uint8_t * scales_base = src + n_blocks * (Q4_K_QK / 2);
    const uint8_t * dm_base     = scales_base + n_blocks * K_SCALE_SIZE;

    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t gid = item.get_global_id(0);
                if (gid >= n_work_items) return;

                const int64_t start_blk = gid * SBS;
                const int64_t end_blk = std::min(start_blk + (int64_t)SBS, n_blocks);

                // Load all dm values for the thread's blocks (4 bytes per block)
                simd<uint8_t, SBS * 4> dm_all_bytes = 0;
                if (start_blk + SBS <= n_blocks) {
                    dm_all_bytes = block_load<uint8_t, SBS * 4>(dm_base + start_blk * 4, overaligned<4>);
                } else {
                    for (int b = 0; b < (int)(end_blk - start_blk); ++b) {
                        simd<uint8_t, 4> temp = block_load<uint8_t, 4>(dm_base + (start_blk + b) * 4, overaligned<4>);
                        dm_all_bytes.template select<4, 1>(b * 4) = temp;
                    }
                }
                simd<fp16, SBS * 2> dm_all = dm_all_bytes.template bit_cast_view<fp16>();

                for (int64_t blk = start_blk; blk < end_blk; ++blk) {
                    const int b_idx = blk - start_blk;
                    const float d    = (float)dm_all[b_idx * 2].read();
                    const float dmin = (float)dm_all[b_idx * 2 + 1].read();

                    const uint8_t * qs         = qs_base     + blk * (Q4_K_QK / 2);
                    const uint8_t * scales_ptr = scales_base + blk * K_SCALE_SIZE;
                    OT * block_dst             = dst + blk * Q4_K_QK;

                    // Load 12 bytes of scales. Over-load to 16 bytes for block_load efficiency.
                    simd<uint8_t, 16> scales_load = 0;
                    if (blk < n_blocks - 1) {
                        scales_load = block_load<uint8_t, 16>(scales_ptr, overaligned<4>);
                    } else {
                        // Safe load for the last block's scales (12 bytes)
                        #pragma unroll
                        for (int i = 0; i < 3; ++i) {
                            simd<uint8_t, 4> temp = block_load<uint8_t, 4>(scales_ptr + i * 4, overaligned<4>);
                            scales_load.template select<4, 1>(i * 4) = temp;
                        }
                    }
                    simd<uint8_t, 12> scales_data = scales_load.template select<12, 1>(0);

                    #pragma unroll
                    for (int sg = 0; sg < 4; ++sg) {
                        const int j_low = sg * 2;
                        const int j_high = sg * 2 + 1;

                        uint8_t sc_low, m_low, sc_high, m_high;
                        if (j_low < 4) {
                            sc_low = scales_data[j_low] & 63;
                            m_low = scales_data[j_low + 4] & 63;
                        } else {
                            sc_low = (scales_data[j_low + 4] & 0xF) | ((scales_data[j_low - 4] >> 2) & 0x30);
                            m_low = (scales_data[j_low + 4] >> 4) | ((scales_data[j_low] >> 2) & 0x30);
                        }
                        if (j_high < 4) {
                            sc_high = scales_data[j_high] & 63;
                            m_high = scales_data[j_high + 4] & 63;
                        } else {
                            sc_high = (scales_data[j_high + 4] & 0xF) | ((scales_data[j_high - 4] >> 2) & 0x30);
                            m_high = (scales_data[j_high + 4] >> 4) | ((scales_data[j_high] >> 2) & 0x30);
                        }

                        float d_sc_low  = d    * float(sc_low);
                        float dm_m_low  = dmin * float(m_low);
                        float d_sc_high = d    * float(sc_high);
                        float dm_m_high = dmin * float(m_high);

                        // Contiguous block load for qs
                        simd<uint8_t, 32> packed = block_load<uint8_t, 32>(qs + sg * 32, overaligned<32>);
                        simd<uint8_t, 32> low_nibbles = packed & (uint8_t)0x0F;
                        simd<uint8_t, 32> high_nibbles = packed >> 4;

                        // Vectorized math
                        simd<float, 32> result_low  = d_sc_low  * convert<float>(low_nibbles)  - dm_m_low;
                        simd<float, 32> result_high = d_sc_high * convert<float>(high_nibbles) - dm_m_high;

                        OT* out_low = block_dst + j_low * 32;
                        OT* out_high = block_dst + j_high * 32;

                        if constexpr (std::is_same_v<OT, fp16>) {
                            simd<fp16, 32> hf_low = result_low, hf_high = result_high;
                            block_store<fp16, 32>(reinterpret_cast<fp16*>(out_low), hf_low);
                            block_store<fp16, 32>(reinterpret_cast<fp16*>(out_high), hf_high);
                        }
#ifdef GGML_SYCL_HAS_BF16
                        else if constexpr (std::is_same_v<OT, bf16>) {
                            simd<bf16, 32> bf_low = result_low, bf_high = result_high;
                            block_store<bf16, 32>(reinterpret_cast<bf16*>(out_low), bf_low);
                            block_store<bf16, 32>(reinterpret_cast<bf16*>(out_high), bf_high);
                        }
#endif
                        else {
                            block_store<float, 32>(out_low, result_low);
                            block_store<float, 32>(out_high, result_high);
                        }
                    }
                }
            }
        );
    };

    stream->submit(cgf);
}

} // namespace gguf
} // namespace omni_xpu

#endif // GGML_SYCL_DEQUANTIZE_ESIMD_HPP
