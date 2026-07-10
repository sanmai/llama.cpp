#!/usr/bin/env python3
"""Validate two-level (canonical / N4_K) NVFP4 scaling in a GGUF.

Based on scripts/nvfp4-subnormal-stats.py. Where that script bounds the ceiling
of a *single* global lift, this one validates that a real per-tensor s_global
(.scale, NVIDIA's recipe s_global = global_amax/(448*6)) has spread the per-16
UE4M3 block scales across E4M3's normal range. For each NVFP4 weight tensor it
reports:
  - UE4M3 block-scale subnormal fraction (should be ~0 once s_global lifts the
    block scales out of the subnormal floor),
  - the octave span of the decoded per-16 block scales (confirms they VARY
    rather than collapsing onto the ~7 subnormal rungs),
  - the sibling per-tensor .scale (s_global) value.
Aggregate: block-scale distribution, distinct codes used, and how s_global
varies across tensors.
"""

import sys

import numpy as np
from gguf import GGUFReader, GGMLQuantizationType


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

    # sibling per-tensor .scale (s_global) values, keyed by the weight base name
    scale_vals = {}
    for t in reader.tensors:
        if t.name.endswith(".scale"):
            scale_vals[t.name[:-len(".scale")]] = float(np.asarray(t.data).reshape(-1)[0])

    all_codes = []
    s_globals = []
    print(f"{'tensor':<40} {'codes':>9} {'subn%':>6} {'blk_min':>9} {'blk_max':>9} {'oct':>5} {'s_global':>11}")
    for t in reader.tensors:
        if t.tensor_type != GGMLQuantizationType.NVFP4:
            continue
        blocks = np.ascontiguousarray(t.data).reshape(-1, 36)
        codes = blocks[:, :4].reshape(-1)
        all_codes.append(codes)

        sub = np.count_nonzero((codes >= 1) & (codes <= 7))
        nz = lut[codes][lut[codes] > 0]
        octaves = np.log2(nz.max() / nz.min()) if nz.size else 0.0
        base = t.name[:-len(".weight")] if t.name.endswith(".weight") else t.name
        sg = scale_vals.get(base, float("nan"))
        if not np.isnan(sg):
            s_globals.append(sg)
        print(f"{base:<40} {codes.size:>9} {100*sub/codes.size:>5.1f}% "
              f"{nz.min():>9.2e} {nz.max():>9.2e} {octaves:>5.1f} {sg:>11.3e}")

    codes = np.concatenate(all_codes)
    nz = lut[codes][lut[codes] > 0]
    sub = np.count_nonzero((codes >= 1) & (codes <= 7))
    p = np.percentile(nz, [0.1, 1, 50, 99, 99.9])

    print("\n=== aggregate over all NVFP4 tensors ===")
    print(f"total micro-scale codes : {codes.size}")
    print(f"subnormal codes (1..7)  : {sub}  ({100*sub/codes.size:.2f}%)")
    print(f"block scale min_nz..max : {nz.min():.3e} .. {nz.max():.3e}  ({np.log2(nz.max()/nz.min()):.1f} octaves)")
    print(f"block scale pct [0.1,1,50,99,99.9] : " + ", ".join(f"{v:.3e}" for v in p))
    print(f"distinct UE4M3 codes used : {np.unique(codes).size} / 254")

    if s_globals:
        sg = np.array(s_globals)
        print(f"\nper-tensor s_global (.scale) over {sg.size} tensors:")
        print(f"  range {sg.min():.3e} .. {sg.max():.3e}  ({np.log2(sg.max()/sg.min()):.1f} octaves), median {np.median(sg):.3e}")

    subn_pct = 100 * sub / codes.size
    span_oct = np.log2(nz.max() / nz.min())
    vary = subn_pct < 5.0 and span_oct > 2.0
    print(f"\nverdict: block scales {'VARY (normal-range two-level)' if vary else 'still pinned (subnormal/single-level)'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3.6-27b-n4k.gguf")
