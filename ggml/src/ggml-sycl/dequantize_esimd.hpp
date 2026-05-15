// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2025 Intel Corporation
//
// Q4_0 dequantization via Intel ESIMD.
// Ported from llm-scaler (omni_xpu_kernel/csrc/gguf_dequant.cpp).

#ifndef GGML_SYCL_DEQUANTIZE_ESIMD_HPP
#define GGML_SYCL_DEQUANTIZE_ESIMD_HPP

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "common.hpp"

namespace ggml_sycl_esimd {

namespace esimd = sycl::ext::intel::esimd;

// Each work item handles Q4_0_SBS blocks; Q4_0_WG_SIZE work items per work group.
constexpr int Q4_0_SBS     = 16;
constexpr int Q4_0_WG_SIZE = 64;

// Size guard: below this block count, per-call ESIMD overhead (kernel JIT,
// submit latency, register-context setup) dominates over its per-element
// speedup. Per-token decode ops (embedding lookup, K/V conversion, residual
// mul_mat fallbacks) typically sit well below this; prefill weight dequants
// on a 7B model sit at ~hundreds of thousands of blocks per call.
constexpr int Q4_0_MIN_BLOCKS = 1024;

// Q4_0 ESIMD kernel
// Output layout: [0-15] = low nibbles, [16-31] = high nibbles
//
// parallel_for is kept inside the kernel function so the ESIMD lambda is
// emitted at template instantiation. Pulling parallel_for out to the
// launcher (with the kernel called from inside the lambda) measurably
// costs the kernel-level speedup, presumably because the SYCL compiler
// doesn't inline through that boundary the same way.
template <typename OT>
void dequantize_block_q4_0_esimd(
    const uint8_t * __restrict__ src,
    OT * __restrict__ dst,
    const int64_t n_blocks,
    sycl::handler & h) {

    const int64_t n_work_items = (n_blocks + Q4_0_SBS - 1) / Q4_0_SBS;
    const int64_t global_size  = (n_work_items + Q4_0_WG_SIZE - 1) / Q4_0_WG_SIZE * Q4_0_WG_SIZE;

    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(Q4_0_WG_SIZE)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int64_t gid = item.get_global_id(0);
            if (gid >= n_work_items) return;

            const int64_t start_block = gid * Q4_0_SBS;
            const int64_t end_block   = std::min(start_block + Q4_0_SBS, n_blocks);

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

// Q4_0 ESIMD kernel for the reorder layout: [all qs (n_blocks * 16 bytes)]
// followed by [all d values (n_blocks * sizeof(ggml_half))]. Used when the
// SYCL optimizer has physically repacked the weight tensor at load time.
// The unpack arithmetic is identical to the standard-layout kernel; only
// the source pointer math differs.
template <typename OT>
void dequantize_block_q4_0_reorder_esimd(
    const uint8_t * __restrict__ src,
    OT * __restrict__ dst,
    const int64_t n_blocks,
    sycl::handler & h) {

    const int64_t n_work_items = (n_blocks + Q4_0_SBS - 1) / Q4_0_SBS;
    const int64_t global_size  = (n_work_items + Q4_0_WG_SIZE - 1) / Q4_0_WG_SIZE * Q4_0_WG_SIZE;

    const uint8_t *    qs_base = src;
    const sycl::half * d_base  = reinterpret_cast<const sycl::half *>(src + n_blocks * (QK4_0 / 2));

    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(Q4_0_WG_SIZE)),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
            const int64_t gid = item.get_global_id(0);
            if (gid >= n_work_items) return;

            const int64_t start_block = gid * Q4_0_SBS;
            const int64_t end_block   = std::min(start_block + Q4_0_SBS, n_blocks);

            esimd::simd<uint32_t, 16> offsets;
            #pragma unroll
            for (int i = 0; i < 16; ++i) offsets[i] = i;

            for (int64_t blk = start_block; blk < end_block; ++blk) {
                const uint8_t *  block_qs = qs_base + blk * (QK4_0 / 2);
                const sycl::half scale    = d_base[blk];
                OT * block_dst = dst + blk * QK4_0;

                esimd::simd<uint8_t, 16> packed =
                    esimd::gather<uint8_t, 16>(block_qs, offsets);

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

} // namespace ggml_sycl_esimd

#endif // GGML_SYCL_DEQUANTIZE_ESIMD_HPP
