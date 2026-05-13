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

} // namespace gguf
} // namespace omni_xpu

#endif // GGML_SYCL_DEQUANTIZE_ESIMD_HPP
