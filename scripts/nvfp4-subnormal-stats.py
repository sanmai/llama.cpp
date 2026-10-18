#!/usr/bin/env python3
"""Report UE4M3 subnormal-scale fraction in an NVFP4 GGUF.

ggml's NVFP4 block is 36 bytes for 64 weights: d[4] (UE4M3 micro-scales, one per
16-element sub-block) followed by qs[32] (packed E2M1 codes). A UE4M3 code is
*subnormal* when its 4 exponent bits are zero (codes 0x01..0x07), where the scale
grid degrades from 3-mantissa-bit floating to coarse fixed 2^-10 spacing. A
per-tensor / per-channel second scale can only rescue these blocks (range
placement); this script measures how many there are, which bounds the ceiling.
"""

import sys

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType

NORMAL_MIN = 2.0 ** -7  # 7.8125e-3, smallest normal decoded UE4M3 scale
UE4M3_MAX  = 224.0      # largest decoded UE4M3 scale (code 0x7E)


def ue4m3_lut():
    lut = np.zeros(256, dtype=np.float64)
    for x in range(256):
        if x == 0 or x == 0x7F:
            continue
        exp = (x >> 3) & 0xF
        man = x & 0x7
        raw = man * 2.0 ** -9 if exp == 0 else (1.0 + man / 8.0) * 2.0 ** (exp - 7)
        lut[x] = raw * 0.5
    return lut


def main(path):
    lut = ue4m3_lut()
    reader = GGUFReader(path)

    all_codes = []
    print(f"{'tensor':<48} {'codes':>10} {'subnorm%':>9} {'min_nz':>10} {'max':>9}")
    for t in reader.tensors:
        if t.tensor_type != GGMLQuantizationType.NVFP4:
            continue
        blocks = np.ascontiguousarray(t.data).reshape(-1, 36)
        codes = blocks[:, :4].reshape(-1)
        all_codes.append(codes)

        sub = np.count_nonzero((codes >= 1) & (codes <= 7))
        scales = lut[codes]
        nz = scales[scales > 0]
        print(f"{t.name:<48} {codes.size:>10} {100*sub/codes.size:>8.2f}% "
              f"{nz.min():>10.2e} {nz.max():>9.2e}")

    codes = np.concatenate(all_codes)
    scales = lut[codes]
    nz = scales[scales > 0]
    sub = np.count_nonzero((codes >= 1) & (codes <= 7))
    p = np.percentile(nz, [0.1, 1, 50, 99, 99.9])

    print("\n=== aggregate over all NVFP4 tensors ===")
    print(f"total micro-scale codes : {codes.size}")
    print(f"subnormal codes (1..7)  : {sub}  ({100*sub/codes.size:.2f}%)")
    print(f"decoded scale min_nz    : {nz.min():.3e}")
    print(f"decoded scale max       : {nz.max():.3e}")
    print(f"decoded scale pct [0.1,1,50,99,99.9] : "
          + ", ".join(f"{v:.3e}" for v in p))

    # N must lift the smallest *meaningful* scale (p0.1) above NORMAL_MIN, while
    # keeping the largest scale below UE4M3_MAX after *2^N.
    n_lift = np.log2(NORMAL_MIN / p[0])           # to rescue p0.1
    n_sat  = np.log2(UE4M3_MAX / nz.max())        # ceiling before saturation
    print(f"\nN to rescue p0.1 scale : >= {n_lift:.2f}")
    print(f"N before saturation    : <  {n_sat:.2f}")
    feasible = n_sat > n_lift
    print(f"single global N feasible: {feasible}"
          + ("" if feasible else "  (block-scale span exceeds UE4M3 normal range)"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-27b-nvfp4.gguf")
