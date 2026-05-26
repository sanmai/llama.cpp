// Unit test for the NVFP4 micro-scale MSE search (best_scale_nvfp4 in ggml-quants.c).
//
// The search runs per 16-element sub-block: starting from the naive scale amax/6 it
// also tries the two adjacent UE4M3 codes and keeps whichever minimizes the squared
// reconstruction error. It is reached through the exported quantize_row_nvfp4_ref(),
// which stores the selected scale code in block_nvfp4::d[].
//
// We re-derive that argmin here with a faithful copy of the inner reconstruction and
// assert the stored code matches it, then confirm the offset search actually fires -
// otherwise the {-1, +1} candidates would be dead code.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#undef NDEBUG
#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// nearest E2M1 code for x at scale e -- a verbatim mirror of best_index_mxfp4(),
// kept expression-for-expression so FP contraction matches the implementation.
static int nvfp4_best_index(float x, float e) {
    int   best_index = 0;
    float best_err   = fabsf(kvalues_mxfp4[0]*e - x);
    for (int i = 1; i < 16; i++) {
        float err = fabsf(kvalues_mxfp4[i]*e - x);
        if (err < best_err) {
            best_index = i;
            best_err   = err;
        }
    }
    return best_index;
}

// squared reconstruction error of a sub-block quantized with scale code `code`
static float subblock_sse(const float * xb, int n, int code) {
    const float d = ggml_ue4m3_to_fp32((uint8_t) code);

    float sse = 0.0f;
    for (int j = 0; j < n; j++) {
        const int   l    = nvfp4_best_index(xb[j], d);
        const float diff = xb[j] - d*kvalues_mxfp4[l];
        sse += diff*diff;
    }
    return sse;
}

// independent re-derivation of best_scale_nvfp4(); also reports the naive amax/6 code
static int ref_best_scale(const float * xb, int n, int * first_code) {
    static const int try_offsets[3] = { 0, -1, 1 };

    float amax = 0.0f;
    for (int j = 0; j < n; j++) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }

    const int first = (int) ggml_fp32_to_ue4m3(amax / 6.0f);
    *first_code = first;

    float best_sse  = FLT_MAX;
    int   best_code = 0;
    for (int t = 0; t < 3; t++) {
        const int code = first + try_offsets[t];
        if (code < 0 || code > 0x7E) {
            continue;
        }
        const float sse = subblock_sse(xb, n, code);
        if (sse < best_sse) {
            best_sse  = sse;
            best_code = code;
        }
    }
    return best_code;
}

// deterministic LCG so the test is reproducible without pulling in <random>
static uint32_t rng_state = 1234567u;
static float frand(void) {
    rng_state = rng_state*1664525u + 1013904223u;
    return (float) (rng_state >> 8) / (float) (1u << 24); // [0, 1)
}

int main(void) {
    const int n_sub = QK_NVFP4 / QK_NVFP4_SUB;

    int improvements = 0;
    int total        = 0;

    for (int b = 0; b < 4096; b++) {
        // sweep the per-block magnitude so amax/6 lands across the whole UE4M3
        // range, exercising both the underflow (code 0) and saturation (0x7E) ends
        const float scale = ldexpf(1.0f, (b % 24) - 12); // 2^-12 .. 2^11

        float x[QK_NVFP4];
        for (int i = 0; i < QK_NVFP4; i++) {
            x[i] = (2.0f*frand() - 1.0f) * scale;
        }

        block_nvfp4 y;
        memset(&y, 0, sizeof(y));
        quantize_row_nvfp4_ref(x, &y, QK_NVFP4);

        for (int s = 0; s < n_sub; s++) {
            const float * xb = x + s*QK_NVFP4_SUB;

            int       first    = 0;
            const int expected = ref_best_scale(xb, QK_NVFP4_SUB, &first);

            // the quantizer must store the SSE-minimizing scale code
            assert(y.d[s] == (uint8_t) expected);

            improvements += (y.d[s] != (uint8_t) first);
            total        += 1;
        }
    }

    // the offset search must improve on the naive amax/6 baseline at least sometimes,
    // otherwise the {-1, +1} candidates carry no weight
    assert(improvements > 0);

    printf("test-quantize-nvfp4: %d sub-blocks, search beat amax/6 on %d (%.1f%%) -- OK\n",
           total, improvements, 100.0*improvements/total);
    return 0;
}
