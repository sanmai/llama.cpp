// Unit test for the NVFP4 micro-scale MSE search (best_scale_nvfp4 in ggml-quants.c).
//
// Per 16-element sub-block the quantizer starts from the naive scale amax/6, also tries
// the two neighbouring UE4M3 codes, and keeps whichever gives the smallest squared
// reconstruction error. It is reached through the exported quantize_row_nvfp4_ref(),
// which stores the chosen scale code in block_nvfp4::d[].
//
// Each fixture is a fixed sub-block plus the scale code the search must store. The test
// prints the same per-code error breakdown the search weighs, so every outcome can be
// followed from the trace alone, e.g. for the "ramp" 1..16:
//
//     [ramp ] amax/6 -> code 67   stored 66
//         code  66  d=1.25   sse=14.125    <- chosen
//         code  67  d=1.375  sse=18.0469   baseline
//         code  68  d=1.5    sse=23
//
// i.e. a slightly smaller scale (66) fits the bulk better than the naive amax/6 (67),
// so the search drops one code. The expected codes were captured from this trace;
// dropping an offset or flipping the comparison shifts a stored code and trips an assert.

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#undef NDEBUG
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// nearest E2M1 code for x at scale e -- a verbatim mirror of best_index_mxfp4(),
// kept expression-for-expression so the printed errors match the search's own.
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

// squared reconstruction error of a sub-block quantized with scale `code`
static float subblock_sse(const float * xb, int code) {
    const float d = ggml_ue4m3_to_fp32((uint8_t) code);

    float sse = 0.0f;
    for (int j = 0; j < QK_NVFP4_SUB; j++) {
        const int   l    = nvfp4_best_index(xb[j], d);
        const float diff = xb[j] - d*kvalues_mxfp4[l];
        sse += diff*diff;
    }
    return sse;
}

// the naive scale code the search starts from: amax/6 rounded to UE4M3
static int naive_scale_code(const float * xb) {
    float amax = 0.0f;
    for (int j = 0; j < QK_NVFP4_SUB; j++) {
        amax = fmaxf(amax, fabsf(xb[j]));
    }
    return (int) ggml_fp32_to_ue4m3(amax / 6.0f);
}

struct fixture {
    const char * name;
    int          expected;          // UE4M3 scale code the search must store
    float        xb[QK_NVFP4_SUB];
};

static const struct fixture fixtures[] = {
    // search drops one code: a coarser top, but a far better fit for the bulk
    { "ramp",  66, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },

    // baseline already optimal: both neighbouring codes score worse
    { "geo",   67, { 16, 8, 4, 2, 1, 0.5f, 0.25f, 0.125f,
                     16, 8, 4, 2, 1, 0.5f, 0.25f, 0.125f } },

    // all zero -> code 0; the code-1 candidate is below the UE4M3 range and is skipped
    { "zeros",  0, { 0 } },

    // amax/6 saturates UE4M3 -> code 0x7E; the code+1 candidate is above range, skipped
    { "sat", 0x7E, { 2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000,
                     2000, 2000, 2000, 2000, 2000, 2000, 2000, 2000 } },
};

// quantize a block built from four copies of the fixture, trace the search, and assert
static int run_fixture(const struct fixture * f) {
    float x[QK_NVFP4];
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; s++) {
        memcpy(x + s*QK_NVFP4_SUB, f->xb, sizeof(f->xb));
    }
    block_nvfp4 y;
    memset(&y, 0, sizeof(y));
    quantize_row_nvfp4_ref(x, &y, QK_NVFP4);

    const int first  = naive_scale_code(f->xb);
    const int stored = y.d[0];

    printf("[%-5s] amax/6 -> code %d   stored %d\n", f->name, first, stored);
    for (int off = -1; off <= 1; off++) {
        const int code = first + off;
        if (code < 0 || code > 0x7E) {
            printf("    code %3d  (out of UE4M3 range, skipped)\n", code);
            continue;
        }
        printf("    code %3d  d=%-9.6g sse=%-11.6g%s%s\n",
               code, ggml_ue4m3_to_fp32((uint8_t) code), subblock_sse(f->xb, code),
               code == first  ? "baseline" : "",
               code == stored ? "  <- chosen" : "");
    }

    // every sub-block is a copy, so all four stored scales must equal the winner
    for (int s = 0; s < QK_NVFP4 / QK_NVFP4_SUB; s++) {
        assert(y.d[s] == (uint8_t) f->expected);
    }
    return stored != first; // moved off the naive scale?
}

int main(void) {
    const size_t n = sizeof(fixtures) / sizeof(fixtures[0]);

    int moved = 0;
    for (size_t i = 0; i < n; i++) {
        moved += run_fixture(&fixtures[i]);
    }

    // at least one fixture must pick an offset, else the {-1, +1} search is dead code
    assert(moved > 0);

    printf("test-quantize-nvfp4: %zu fixtures, %d moved off amax/6 -- OK\n", n, moved);
    return 0;
}
