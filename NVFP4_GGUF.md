# NVFP4 weight quantization: MSE scale-search

## Hypothesis

The per-sub-block MSE scale search added to `quantize_row_nvfp4_ref` (commit `619f99fef`)
produces strictly better NVFP4 *weight* quantization than the single-code baseline.

Concretely: Qwen3-1.7B-Base quantized to NVFP4 **with** the search will have lower
KL-divergence from its f16 reference (and lower perplexity) than the same model quantized
**without** it.

- Prior signal: `test-quantize-fns -v` nvfp4 dot-product error dropped 0.01977 -> 0.00242 (~8x)
  and absolute error 0.00234 -> 0.00203 when the search was added. We expect this to propagate
  to a clear end-to-end gain.
- Bound: the search only refines each 16-element sub-block's UE4M3 scale; it cannot restore
  NVFP4's dropped per-tensor scale. So it should *narrow* the f16->nvfp4 gap, not close it.
- Null hypothesis: searched KLD/PPL ~= unsearched (the synthetic gain does not survive on real
  weights).

## Setup

- Model: Qwen/Qwen3-1.7B-Base (dense), converted to f16 GGUF as the reference.
- Corpus: `wikitext-2-raw/wiki.test.raw`, ctx 512, `-b 4096 -ub 2048`.
- Base logits: `kld-base-qwen3.dat` (from the f16 model).
- A/B: `nvfp4-new` = HEAD (search), `nvfp4-old` = HEAD~1 (single-code), `q4_0` = same-bpw context.
- Quantizer change is `quantize_row_nvfp4_ref` only; `llama-quantize` NVFP4 selectability added
  separately and present in both arms. CPU-side quantization; imatrix not used (constant across A/B).

## Honest framing

- This measures ggml's single-level NVFP4 (old vs new search), not NVIDIA's two-level PTQ NVFP4.
- `quantize_row_nvfp4_ref` ignores the imatrix.

## Results

Qwen3-1.7B-Base, full `wiki.test.raw`, ctx 512, b4096/ub2048. KLD/PPL vs the f16 base
(PPL(f16) ~= 9.87). All three quant arms are 4.87 bpw (identical GGUF size, 999.90 MiB):
nvfp4 tensors + Q6_K output + Q8_0 embd, same mix in every arm.

| quant                  | Mean PPL(Q)        | PPL ratio | Mean KLD            | Same top-p |
|------------------------|--------------------|-----------|---------------------|------------|
| nvfp4-old (no search)  | 11.790 +/- 0.086   | 1.19411   | 0.19044 +/- 0.00081 | 78.49%     |
| nvfp4-new (MSE search) | 11.583 +/- 0.082   | 1.17312   | 0.17393 +/- 0.00069 | 78.62%     |
| q4_0 (context)         | 11.387 +/- 0.081   | 1.15325   | 0.16130 +/- 0.00046 | 79.20%     |

**Hypothesis CONFIRMED.** The MSE scale search lowers NVFP4 weight-quant divergence:
Mean KLD 0.19044 -> 0.17393 (**-8.7%**), PPL penalty +19.4% -> +17.3%. The gap is ~20 sigma
on the error bars - robustly real, not noise.

**But the gain is modest, and bounded as predicted.** The ~8x synthetic dot-error improvement
(`test-quantize-fns`, uniform random data) does NOT translate to ~8x end-to-end: real weights
see ~9% lower KLD. The search refines each sub-block's UE4M3 scale but cannot restore NVFP4's
dropped per-tensor scale.

**nvfp4 still trails q4_0 at equal footprint** (searched-nvfp4 0.17393 vs q4_0 0.16130, ~8%
worse KLD). So the search makes nvfp4 *better than before*, not *better than q4_0* - consistent
with the KV-cache finding: ggml's single-level nvfp4 tracks q4_0 and never beats it.

Same-top-p (top-1 token agreement vs the f16 base) tells the same story at coarser resolution:
q4_0 79.20% vs nvfp4 78.62%, a **~0.58 pp** gap in q4_0's favor (the search narrows it from
~0.71 pp at no-search nvfp4 = 78.49%, but does not overtake). Same-top-p only counts the argmax;
the full-distribution shift shows up larger in KLD (the ~8% above).

**Takeaway.** The MSE search is a worthwhile, free improvement to the *existing* NVFP4
reference quantizer (~9% lower KLD for any nvfp4 weight conversion or KV write, and it removed
CPU/CUDA encoding drift). It does not make nvfp4 the preferred 4-bit format. Caveats: single
small model; no imatrix (nvfp4 ignores it - q4_0/q4_K with imatrix would widen their lead).

## Search depth: +/-0 .. +/-3 (Qwen3-1.7B, vs same f16 base)

| search depth        | Mean KLD            | note          |
|---------------------|---------------------|---------------|
| +/-0 (no search)    | 0.190443            | baseline      |
| +/-1                | 0.178935 +/- 0.00070 | +0.0050 (~7s) worse than +/-2 |
| +/-2 (committed)    | 0.173927 +/- 0.00069 | **minimum**   |
| +/-3                | 0.177830 +/- 0.00069 | +0.0039 (~5.7s) worse than +/-2 |

A clean valley with the floor at **+/-2** - both under-refining (+/-1, too few candidates) and
over-refining (+/-3) lose, each ~5-7 sigma worse. The +/-3 direction is the informative one: a
min-error search can only *lower* per-block SSE with more candidates, so +/-3 has lower
unweighted SSE yet HIGHER KLD - it optimizes the wrong objective. Unweighted SSE over-weights
the many small values, so a wider scan drops the scale to fit the bulk while distorting the few
large values that drive the output. The committed +/-2 is empirically optimal among the four.
Depth is settled; the only principled lever left is importance-weighting (imatrix).

## x^2-weighted error (+/-2, Qwen3-1.7B, vs same f16 base)

| error metric            | Mean KLD            | PPL ratio | Same top-p |
|-------------------------|---------------------|-----------|------------|
| unweighted (committed)  | 0.173927 +/- 0.00069 | 1.17312   | 78.62%     |
| x^2-weighted            | 0.177634 +/- 0.00070 | 1.17608   | 78.19%     |

x^2-weighting is **worse** by +0.0037 KLD (~5.3 sigma), worse on all three metrics. Hypothesis
refuted. Reconciling with the +/-3 result: unweighted +/-2 is a balance point. +/-3 (more
candidates -> picks a lower scale, fits the small-value bulk harder) pushes *toward small
values* and hurt; x^2-weighting pushes *toward the few large values* (the max, which amax/6
already pins) and also hurt. Both directions away from the committed unweighted +/-2 regress.

The lesson: per-sub-block reconstruction error - weighted or not - is only a proxy for KLD, and
unweighted +/-2 happens to be the best proxy among these variants. A crude per-weight reweight
(x^2) is just a differently-misaligned proxy. The principled lever is the **imatrix**
(activation-aware importance, what make_qx_quants uses as `qw`), which `quantize_row_nvfp4_ref`
currently ignores even though `quantize_nvfp4` receives `quant_weights`. That is a real,
larger change; depth and x^2 are dead ends.

## Verdict on tuning the search

Committed unweighted +/-2 is the sweet spot among {+/-2, +/-3, x^2-weighted +/-2}. Keep it.
Real further gains require imatrix plumbing, not search-depth or naive reweighting.

## imatrix (Qwen3-1.7B, calibrated on wiki.train, KLD vs same f16 base)

Plumbed the imatrix into nvfp4: `quantize_nvfp4` now branches on `quant_weights`, and
`quantize_row_nvfp4_impl` weights the +/-2 UE4M3 scale search by `qw*sqrt(sigma2+x^2)` (same
form as `quantize_row_q4_0_impl`). q4_0 re-quantized with the same imatrix (overwrites the
no-imatrix q4_0).

|       | no-imatrix KLD | imatrix KLD | imatrix gain |
|-------|----------------|-------------|--------------|
| q4_0  | 0.16130        | 0.10145     | -37.1%       |
| nvfp4 | 0.17393        | 0.14431     | -17.0%       |

The nvfp4 imatrix plumbing **works** (-17%, best nvfp4 result; a real keepable improvement,
unlike +/-3 and x^2). But q4_0 exploits the imatrix >2x as well, so the **gap widens**:
nvfp4-imat is 42% worse than q4_0-imat (0.1443 vs 0.1015), up from 8% without imatrix.

Why nvfp4 can't ride it as far (the structural ceiling, sharpened): q4_0's `make_qx_quants`
lets the imatrix steer a *continuous* fp16 scale (fine +/-9x0.1 grid) AND the per-element level
assignment (weighted least-squares). nvfp4's path can only let the imatrix pick among ~5
*discrete* UE4M3 rungs, with per-element E2M1 codes still nearest (fixed non-uniform levels -
no room to spend importance on a value). Coarse discrete scale throttles the benefit.

## Overall conclusion

On quality, q4_0 beats nvfp4 at every level and the imatrix widens the lead - the E2M1 +
per-16 FP8-scale format is the ceiling, not the quantizer. nvfp4's case is purely **prefill
throughput**: native FP4 MMA on Blackwell gives +~19% pp on the 7B (-2% decode). So nvfp4 is a
speed-for-quality trade, not a quality win. The two keepable contributions, independent of that
verdict: (1) the +/-2 MSE scale search (~9% better nvfp4 quant, fixed CPU/CUDA drift), (2) the
imatrix plumbing (-17%). Both improve the existing type; neither makes nvfp4 preferable to q4_0
on quality.

## Performance (RTX 5090, native FP4 vs q4_0)

`llama-bench`, `-ngl 99 -b 4096 -ub 2048 -r 20`. Each pair is the same f16 source quantized to
nvfp4 (committed +/-2 search) and q4_0; identical GGUF size in each pair (so equal bytes/bandwidth).
On Blackwell the nvfp4 weight matmul takes the native FP4 MMA path (`mmq.cu:125` `use_native_fp4`);
q4_0 uses integer DP4A MMQ. Decode (`tg`, batch 1) is a bandwidth-bound GEMV via MMVQ - no tensor
cores on either side.

Qwen3-1.7B (t/s):

| test    | q4_0            | nvfp4           | delta  |
|---------|-----------------|-----------------|--------|
| pp4096  | 23967 +/- 258   | 24649 +/- 333   | +2.8%  |
| pp8192  | 17713 +/- 112   | 18052 +/- 263   | +1.9%  |
| pp16384 | 11175 +/- 355   | 11330 +/- 314   | +1.4%  |
| tg128   | 588.0 +/- 16    | 558.3 +/- 13    | -5.0%  |

Qwen2.5-7B (t/s):

| test    | q4_0            | nvfp4           | delta   |
|---------|-----------------|-----------------|---------|
| pp4096  | 10944 +/- 89    | 13065 +/- 16    | +19.4%  |
| pp8192  | 8518 +/- 15     | 9738 +/- 15     | +14.3%  |
| pp16384 | 5833 +/- 23     | 6394 +/- 12     | +9.6%   |
| tg128   | 281.2 +/- 0.9   | 275.4 +/- 0.7   | -2.1%   |

Qwen3.6-27B (dense Qwen3.5, t/s, **-b 8192 -ub 4096** "full blast", + Q3_K arm):

| test    | q4_0 (4.5bpw)   | nvfp4 (4.5bpw)  | Q3_K (3.4bpw)   | nvfp4 vs q4_0 | nvfp4 vs Q3_K |
|---------|-----------------|-----------------|-----------------|---------------|---------------|
| pp4096  | 3635.6 +/- 25   | 4804.5 +/- 31   | 3048.7 +/- 9    | +32.2%        | +57.6%        |
| pp8192  | 3427.0 +/- 38   | 4564.9 +/- 27   | 2945.0 +/- 10   | +33.2%        | +55.0%        |
| pp16384 | 3051.8 +/- 10   | 3948.5 +/- 5    | 2644.7 +/- 4    | +29.4%        | +49.3%        |
| tg128   | 84.42 +/- 0.13  | 84.08 +/- 0.12  | 85.65 +/- 0.09  | -0.4%         | -1.8%         |

At 27B (full blast) the prefill win **strengthens and flattens**: ~+30% over q4_0 across all
prompt lengths (the FFN body so dominates a 27B that attention dilution barely registers). The
**decode tax vanishes with scale**: -5.0% (1.7B) -> -2.1% (7B) -> -0.4% (27B), as the per-16
FP8/LUT unpack amortizes. And nvfp4 **beats the smaller Q3_K on prefill by +49-58%** despite
Q3_K's fewer bits: prefill is compute-bound, Q3_K runs DP4A K-quant (no tensor cores), so 3.4
bpw is *slower* there; Q3_K only edges decode (+1.8%, bandwidth-bound). Speed landscape: prefill
nvfp4 >> q4_0 > Q3_K; decode all within ~2%. Power pinned ~572/575 W during prefill, so the +30%
prefill ~= ~30% fewer joules/prompt vs q4_0 (and below Q3_K's energy too).

**Prefill: nvfp4 wins, and the win scales with model size** (1.7B ~1-3%, 7B ~10-19%, 27B ~+30%). That
scaling IS the evidence the FP4 cores are doing the work: native FP4 MMA needs large GEMMs to
flex, and a 1.7B's FFN matmuls are too small. The taper with prompt length (+19 -> +14 -> +10%
at 7B) is attention dilution - O(n^2) f16 attention is identical for both arms and grows as a
share of total time, shrinking the weight-matmul fraction the format difference rides on.

**Decode: nvfp4 loses** (~2% at 7B, ~5% at 1.7B). Same bytes -> same bandwidth; the loss is the
extra per-byte unpack in the GEMV - non-uniform E2M1 LUT lookup (`get_int_from_table_16`) plus a
per-16 UE4M3/FP8 scale decode - that q4_0 avoids (uniform codes straight into DP4A, one fp16
scale per 32). The FP4 MMA that wins prefill is absent on this path, so the format's richness is
pure overhead here. It's the exact mirror of the prefill win.

**Energy:** at ~equal power, +16% prefill t/s (pp4096, 7B) ~= ~16% fewer joules for the prefill
phase; decode ~neutral. So for prefill-of-long-prompts, nvfp4 is faster and cooler. (Power not
yet sampled; this is the speed->energy floor, not a measured J/token.)

**This gain is NOT free - it is a trade, not a Pareto win.** Pair it with the quality result
above: nvfp4 carries ~8% worse KLD than q4_0 at equal footprint. So nvfp4 on Blackwell buys
**+10-19% prefill throughput at the cost of ~8% KLD / ~0.58 pp top-1**, same size. Worth it if
prefill-/energy-bound and tolerant of a small quality dip; q4_0 still wins quality-per-byte. The
imatrix asymmetry tilts the trade further toward q4 (it would widen q4's quality lead while
nvfp4's speed edge persists).

**Native FP4 confirmed three independent ways (no ambiguity):**
- *Compiled* - `cuobjdump --dump-sass libggml-cuda.so` carries `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`
  (= NVFP4: m16n8**k64** tile, E2M1 4-bit operands, UE4M3 block scale, scale_vec::4X; the `.E8`
  sibling is MXFP4). That is `mma.sync...kind::mxf4nvf4` lowered to SASS.
- *Dispatched* - `nsys` on an nvfp4 prefill shows `quantize_mmq_nvfp4` (the FP4 activation packer,
  unique to the `use_native_fp4` branch) + `mul_mat_q<(ggml_type)40,...>` (NVFP4) as the top kernels.
- *Executed* - `ncu` profiles `mul_mat_q<40>` on CC 12.0 (RTX 5090).

**Mechanism of the prefill win (from the SASS):** q4_0's MMQ runs `IMMA.16832.S8.S8` (int8,
**k=32**); nvfp4 runs `OMMA.16864...` (FP4, **k=64**) - double the K-depth per tensor-core
instruction at the same issue slots. That k32->k64 doubling is the architectural source of the
+10-19%, not a vague "FP4 is faster". (ncu also showed the kernel only ~17% SM-busy at pp256 -
the FP4 cores idle at small sizes, which is why the win scales with size/batch -> the batched
avenue below.)

Caveats: clocks unlocked, but 7B sigmas are ~0.1-0.8% (runs long enough for stable boost).
Same-model quality is below.

## 7B quality (Qwen2.5-7B) + the same-model verdict

Weight-quant KLD/PPL vs the qwen2.5-7b f16 base (full `wiki.test.raw`, ctx 512). nvfp4 and
q4_0-fresh are the same ~4.13 GiB; q4_0-imat is +0.3% (imatrix nudges a couple tensor types).

| quant            | PPL ratio | Mean KLD            | Same top-p |
|------------------|-----------|---------------------|------------|
| nvfp4 (search)   | +7.7%     | 0.10524             | 85.21%     |
| nvfp4 + imatrix  | +7.1%     | 0.09235             | 86.33%     |
| q4_0 (no imat)   | +4.1%     | 0.05779             | 88.83%     |
| q4_0 + imatrix   | +1.8%     | 0.04649             | 89.92%     |

- **nvfp4 vs plain q4_0: ~82% worse KLD** (0.105 vs 0.058), ~1.9x the PPL penalty, -3.6 pp
  top-1. At 1.7B (Qwen3) the gap was only ~8%; at 7B (Qwen2.5) it blows open.
- **imatrix helps both, unevenly.** q4_0 halves its PPL penalty (+4.1% -> +1.8%, -20% KLD);
  nvfp4 - once plumbed (`quantize_row_nvfp4_impl`) - gains -12% KLD (0.1052 -> 0.0924). nvfp4
  *can* use the imatrix now; it just rides it less far (single-scale ceiling, see below).
- **nvfp4+imatrix vs the realistic q4_0+imatrix: ~2x worse KLD** (0.0924 vs 0.0465), ~3.9x the
  PPL penalty (+7.1% vs +1.8%), -3.6 pp top-1.

**Same-model speed-vs-quality trade:** nvfp4 buys +10-19% prefill at ~2x worse quantization
quality than q4_0 (imatrix on both: 0.0924 vs 0.0465). Whether that's worth it is
workload-dependent, not a wash: prefill-/throughput-/energy-bound work on consumer Blackwell
(the whole sm_120 line, 5050..5090 + laptop parts) can reasonably take the trade; memory-bound
or quality-per-byte favors q4_0/IQ4_XS. A real trade, not a free win and not a write-off.

**Why 8% (1.7B) -> 82% (7B):** nvfp4's single-level floor. q4_0 rides its fp16-per-32 scale down
to 0.058 on the redundant 7B; nvfp4, missing the per-tensor scale, stalls at 0.105 and cannot
follow. On the harder 1.7B both sit near a ~0.16-0.17 floor and look close. nvfp4 looks
competitive exactly where 4-bit is hard, and falls behind exactly where 4-bit is easy. (Caveat:
1.7B=Qwen3, 7B=Qwen2.5 - same family that showed K-outliers in the KV study - so model and scale
are confounded; the 7B result itself is clean.)

## Quality-per-bit ladder (Qwen2.5-7B, same f16 base, no imatrix unless noted)

Where nvfp4 actually sits among 4-bit-class formats. Sorted by KLD (best first):

| format          | size      | Mean KLD | PPL ratio | Same top-p |
|-----------------|-----------|----------|-----------|------------|
| IQ4_XS          | 3.96 GiB  | 0.04071  | +3.0%     | 90.51%     |
| q4_0 + imatrix  | 4.14 GiB  | 0.04649  | +1.8%     | 89.92%     |
| q4_0            | 4.13 GiB  | 0.05779  | +4.1%     | 88.83%     |
| **nvfp4 + imat**| 4.13 GiB  | 0.09235  | +7.1%     | 86.33%     |
| nvfp4 (search)  | 4.13 GiB  | 0.10524  | +7.7%     | 85.21%     |
| Q3_K_M          | 3.55 GiB  | 0.10925  | +8.6%     | 85.08%     |
| Q3_K_S          | 3.25 GiB  | 0.19372  | +12.2%    | 79.57%     |

**nvfp4 sits at the q3/q4 boundary in quality; the imatrix lifts it but cannot reach real 4-bit.**
- Without imatrix it merely matches Q3_K_M (0.1052 vs 0.1092 KLD) at 0.58 GiB more size. With the
  imatrix it **clears Q3_K_M** (0.0924 vs 0.1092, 86.33% vs 85.08% top-1) - so it's no longer
  q3-tier, but it's still not q4-tier.
- It remains dominated on quality-per-byte by the real 4-bit formats: q4_0 (0.058) and q4_0+imatrix
  (0.046) at the same size, and IQ4_XS (0.041) which is *smaller* (3.96 GiB) **and** ~2x better KLD.

So on quality-per-byte alone, q4_0/IQ4_XS win and nvfp4 is the wrong tool for the memory-bound
"fit a bigger model" case. nvfp4's non-dominated property is **Blackwell native FP4 prefill
throughput** (+10-19%, see Performance) - available across the whole consumer sm_120 line, not
just halo cards. That makes it a genuine trade for prefill-/throughput-/energy-bound workloads:
spend ~2x quantization quality (vs q4_0+imatrix) to buy double-digit prefill on commodity
Blackwell. Worth it or not depends on the workload; it is not a blanket write-off.

## Why the imatrix helps nvfp4 less: the single-scale ceiling

The imatrix's only lever is the **scale choice** - in *both* formats. For a fixed scale `d` the
weighted block error `sum_j w_j (x_j - c_j*d)^2` separates per element, so each code
`c_j = argmin_c w_j (x_j - c*d)^2` = nearest code, independent of `w_j` (a non-negative constant
just scales that term). Importance weights only matter where the error is *summed across j* -
i.e. picking `d`. q4_0's per-element levels are nearest-given-scale too; the imatrix does not
touch per-element rounding in either.

So the binding constraint is purely the **scale's resolution**:
- q4_0: `d` is a **continuous fp16** value (LS-fit via `sumlx/suml2`, finely searched). The
  importance-weighted optimum is a real number and a continuous knob lands on it.
- nvfp4: `d` is a **UE4M3** value; the +/-2 search reaches ~5 of them. The weighted optimum is
  *rounded to the nearest reachable rung*. The imatrix computes a better target; nvfp4 cannot
  represent it. Hence -17% (nvfp4) vs -37% (q4_0).

**Unifying point:** that single coarse scale is exactly ggml dropping real-NVFP4's second-level
**per-tensor FP32 scale** (true NVFP4 effective scale = `FP32_tensor * UE4M3_subblock`; ggml
kept only the UE4M3 micro-scale). This is the *same* omission that makes the block writable
incrementally - i.e. KV-viable. One design choice, three consequences: KV-viability (+), capped
weight quality vs q4_0 (-), and a starved imatrix lever (-).

**Actionable, and it splits weights from KV:**
- **Weights are static** -> two-level scaling is free (it is how real NVFP4 PTQ checkpoints
  store weights). Adding a per-tensor FP32 scale would hand the imatrix a continuous knob and
  very likely recover most of the gap to q4_0 - the real path to "nvfp4 weights competitive
  *and* +19% prefill". It is a format change (extra per-tensor scale), not a quantizer tweak.
- **KV is block-local** -> the cache is written per token, so a per-tensor scale is impossible;
  the coarse ceiling is inherent and the KV verdict (tracks q4_0, never better) is permanent.

One-liner: the imatrix does not generalize *poorly* for nvfp4 - it generalizes exactly as far
as the format's one coarse degree of freedom allows, and that freedom was traded for
KV-viability. Restore the second scale level (weights only) and the imatrix story changes.

(Generalization caveats for the magnitudes: the imatrix was calibrated on wiki.train and
evaluated on wiki.test - same domain, so -17%/-37% are optimistic in-domain bounds; a diverse
calibration set trades some in-domain gain for robustness. The *relative* claim - nvfp4 gains
~half of q4_0 and the gap widens - is structural and should hold across models/domains; the
absolute KLDs and the imatrix deltas are expected to compress on larger models, which the 7B
sweep will show.)

## The UE4M3 micro-scale runs ~100% subnormal (Qwen3.6-27B)

Measured directly from `qwen3.6-27b-nvfp4.gguf` (`scripts/nvfp4-subnormal-stats.py`, over all
1.63e9 micro-scale codes): **99.94% of UE4M3 block scales are subnormal** (codes 0x01..0x07). The
decoded scales are pinned to the 7 subnormal levels - median 2.93e-3, p99.9 = 6.84e-3 (the
*largest* subnormal), max across the whole model only 0.117. The 3-mantissa-bit normal grid is
essentially never used.

Mechanism: encode stores `ue4m3(amax/6)` and decode folds a `*0.5`, so the effective per-16 scale
is `d ~= amax/12`. UE4M3's normal range starts at 2^-7 = 7.81e-3, so a block is normal only when
`amax > 0.094`; typical Qwen weight sub-blocks (median amax ~0.035) fall below that. So the per-16
"FP8 scale" is in practice a **7-level log grid**, far coarser than the ~5 rungs the +/-2 search
suggests above - that, not the search, is the real scale ceiling.

This sharpens the diagnosis: real NVFP4's dropped second scale does **range placement** (its
per-tensor FP32 normalizes block scales to the top of E4M3's range); ggml's omission lands them in
the subnormal floor instead. Because the miscalibration is *uniform across every tensor* (whole-model
scale span 9.77e-4..0.117 = ~7 octaves, well inside UE4M3's ~15-octave normal range), a **single
global exponent shift** rescues it - no per-tensor storage. Feasible N in [3, 10.9]; N=6 centers the
distribution. And the precision is gained at quantize time (scales stored in the normal grid), so the
native FP4 OMMA reads the better scales directly; only one constant epilogue `*2^-N` corrects the
magnitude -> **native-FP4-compatible, no new type**.

Ceiling KLD (no-subnormal requantize vs the 27B f16 base, `--chunks 200`) is being measured next; if
it moves nvfp4 KLD substantially toward q4_0, the cheap global recalibration - not a per-tensor type
- is the fix.

## Future avenue: batched / concurrent-prefill serving

All perf numbers above are single-stream `pp` (+10-19% nvfp4 on 7B). The strongest unmeasured
case for nvfp4 is **batched serving** - many parallel prefills, long-context RAG, agentic
workloads - which is *more* compute-bound, so the native FP4 advantage should grow beyond the
single-stream figure. This is nvfp4's best-case workload and the natural home for the
throughput-bound segment. Native FP4 is the entire consumer Blackwell line (sm_120: 5050..5090),
not a 5090 halo feature, so the segment is mass-market; the +10-19% ratio is a 5090 number that
should roughly transfer down the line but is unmeasured on smaller cards. Worth a llama-bench
(multiple parallel sequences) or server throughput run before a final line on nvfp4's usefulness.
This is orthogonal to the weight-quality avenue above: batched prefill widens nvfp4's *speed*
edge; the two-level-scale restoration would fix its *quality* deficit - independent levers.
