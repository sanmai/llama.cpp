// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Intel Corporation
//
// Q4_0 / Q4_K dequantization via Intel ESIMD.
// Ported from llm-scaler (omni_xpu_kernel/csrc/gguf_dequant.cpp); Q4_0 body
// is verbatim, Q4_K body keeps the source's structure but moves intermediate
// math to fp32 to match GGML's target precision (see Q4_K kernel comment).

#ifndef GGML_SYCL_DEQUANTIZE_ESIMD_HPP
#define GGML_SYCL_DEQUANTIZE_ESIMD_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "common.hpp"

namespace ggml_sycl_esimd {

namespace esimd = sycl::ext::intel::esimd;

// Q4_0 ESIMD kernel (GGML layout)
// Output layout: [0-15] = low nibbles, [16-31] = high nibbles
template <typename OT>
void dequantize_block_q4_0_esimd(
    const uint8_t * __restrict__ src,
    OT * __restrict__ dst,
    const int64_t n_blocks,
    sycl::handler & h) {

    constexpr int SBS     = 16;
    constexpr int WG_SIZE = 64;
    const int64_t n_work_items = (n_blocks + SBS - 1) / SBS;
    const int64_t padded_size  = (n_work_items + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int64_t gid = item.get_global_id(0);
            if (gid >= n_work_items) return;

            const int64_t start_block = gid * SBS;
            const int64_t end_block   = std::min(start_block + SBS, n_blocks);

            esimd::simd<uint32_t, 16> offsets;
            #pragma unroll
            for (int i = 0; i < 16; ++i) offsets[i] = i;

            for (int64_t blk = start_block; blk < end_block; ++blk) {
                const uint8_t * block_src = src + blk * sizeof(block_q4_0);
                OT * block_dst = dst + blk * QK4_0;

                const sycl::half scale = *reinterpret_cast<const sycl::half *>(block_src);
                esimd::simd<uint8_t, 16> packed =
                    esimd::gather<uint8_t, 16>(block_src + sizeof(ggml_half), offsets);

                esimd::simd<uint8_t, QK4_0> unpacked;
                unpacked.select<16, 1>(0)  = packed & (uint8_t)0x0F;
                unpacked.select<16, 1>(16) = packed >> 4;

                esimd::simd<int16_t, QK4_0> signed_vals = unpacked;
                signed_vals = signed_vals - (int16_t)8;
                esimd::simd<sycl::half, QK4_0> result = signed_vals * scale;

                if constexpr (std::is_same_v<OT, sycl::half>) {
                    esimd::block_store<sycl::half, QK4_0>(
                        reinterpret_cast<sycl::half *>(block_dst), result);
                }
#ifdef GGML_SYCL_HAS_BF16
                else if constexpr (std::is_same_v<OT, sycl::ext::oneapi::bfloat16>) {
                    esimd::simd<sycl::ext::oneapi::bfloat16, QK4_0> bf_result = result;
                    esimd::block_store<sycl::ext::oneapi::bfloat16, QK4_0>(
                        reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(block_dst), bf_result);
                }
#endif
                else {
                    esimd::simd<float, QK4_0> f_result = result;
                    esimd::block_store<float, QK4_0>(block_dst, f_result);
                }
            }
        });
}

// Q4_K ESIMD kernel (GGML layout)
//
template <typename OT>
void dequantize_block_q4_K_esimd(
    const uint8_t * __restrict__ src,
    OT * __restrict__ dst,
    const int64_t n_blocks,
    sycl::handler & h) {

    constexpr int WG_SIZE = 64;
    const int64_t padded_size = (n_blocks + WG_SIZE - 1) / WG_SIZE * WG_SIZE;

    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(padded_size), sycl::range<1>(WG_SIZE)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int64_t blk = item.get_global_id(0);
            if (blk >= n_blocks) return;

            const uint8_t * block_src = src + blk * sizeof(block_q4_K);
            OT * block_dst = dst + blk * QK_K;

            const float d    = float(*reinterpret_cast<const sycl::half *>(block_src));
            const float dmin = float(*reinterpret_cast<const sycl::half *>(block_src + sizeof(ggml_half)));

            esimd::simd<uint8_t, K_SCALE_SIZE> scales_data;
            const uint8_t * scales_ptr = block_src + 2 * sizeof(ggml_half);
            #pragma unroll
            for (int i = 0; i < K_SCALE_SIZE; ++i) scales_data[i] = scales_ptr[i];

            const uint8_t * qs = block_src + 2 * sizeof(ggml_half) + K_SCALE_SIZE;

            esimd::simd<uint32_t, 32> offsets32;
            #pragma unroll
            for (int i = 0; i < 32; ++i) offsets32[i] = i;

            // 8 sub-blocks of 32 elements, processed as 4 (low, high) pairs
            #pragma unroll
            for (int sg = 0; sg < 4; ++sg) {
                const int j_low  = sg * 2;
                const int j_high = sg * 2 + 1;

                // 6-bit packed scales/mins unpack (matches GGML get_scale_min_k4)
                uint8_t sc_low, m_low, sc_high, m_high;
                if (j_low < 4) {
                    sc_low = scales_data[j_low]     & 63;
                    m_low  = scales_data[j_low + 4] & 63;
                } else {
                    sc_low = (scales_data[j_low + 4] & 0xF) | ((scales_data[j_low - 4] >> 2) & 0x30);
                    m_low  = (scales_data[j_low + 4] >> 4)  | ((scales_data[j_low]     >> 2) & 0x30);
                }
                if (j_high < 4) {
                    sc_high = scales_data[j_high]     & 63;
                    m_high  = scales_data[j_high + 4] & 63;
                } else {
                    sc_high = (scales_data[j_high + 4] & 0xF) | ((scales_data[j_high - 4] >> 2) & 0x30);
                    m_high  = (scales_data[j_high + 4] >> 4)  | ((scales_data[j_high]     >> 2) & 0x30);
                }

                const float d_sc_low  = d    * float(sc_low);
                const float dm_m_low  = dmin * float(m_low);
                const float d_sc_high = d    * float(sc_high);
                const float dm_m_high = dmin * float(m_high);

                esimd::simd<uint8_t, 32> packed =
                    esimd::gather<uint8_t, 32>(qs + sg * 32, offsets32);
                esimd::simd<uint8_t, 32> low_nibbles  = packed & (uint8_t)0x0F;
                esimd::simd<uint8_t, 32> high_nibbles = packed >> 4;

                esimd::simd<float, 32> result_low, result_high;
                #pragma unroll
                for (int i = 0; i < 32; ++i) {
                    result_low[i]  = d_sc_low  * float(low_nibbles[i])  - dm_m_low;
                    result_high[i] = d_sc_high * float(high_nibbles[i]) - dm_m_high;
                }

                OT * out_low  = block_dst + j_low  * 32;
                OT * out_high = block_dst + j_high * 32;

                if constexpr (std::is_same_v<OT, sycl::half>) {
                    esimd::simd<sycl::half, 32> hf_low  = result_low;
                    esimd::simd<sycl::half, 32> hf_high = result_high;
                    esimd::block_store<sycl::half, 32>(
                        reinterpret_cast<sycl::half *>(out_low),  hf_low);
                    esimd::block_store<sycl::half, 32>(
                        reinterpret_cast<sycl::half *>(out_high), hf_high);
                }
#ifdef GGML_SYCL_HAS_BF16
                else if constexpr (std::is_same_v<OT, sycl::ext::oneapi::bfloat16>) {
                    esimd::simd<sycl::ext::oneapi::bfloat16, 32> bf_low  = result_low;
                    esimd::simd<sycl::ext::oneapi::bfloat16, 32> bf_high = result_high;
                    esimd::block_store<sycl::ext::oneapi::bfloat16, 32>(
                        reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(out_low),  bf_low);
                    esimd::block_store<sycl::ext::oneapi::bfloat16, 32>(
                        reinterpret_cast<sycl::ext::oneapi::bfloat16 *>(out_high), bf_high);
                }
#endif
                else {
                    esimd::block_store<float, 32>(out_low,  result_low);
                    esimd::block_store<float, 32>(out_high, result_high);
                }
            }
        });
}

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_DEQUANTIZE_ESIMD_HPP
