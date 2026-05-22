# N4_0 weight quantization: MSE scale-search

> **Naming.** This format is **N4_0** - single-level NVFP4: a per-16 UE4M3 micro-scale, no
> per-tensor scale. The ggml type is still named `GGML_TYPE_NVFP4` in code (the old name); N4_0
> is the new name and the code refactor is pending. **N4_K** is reserved for a future two-level
> variant (per-tensor FP32 + per-16 UE4M3) - the layout NVIDIA's real NVFP4 PTQ uses. In this
> doc, unqualified **NVFP4** means NVIDIA's two-level reference format; **N4_0** is ggml's
> single-level type.

## Hypothesis

The per-sub-block MSE scale search added to `quantize_row_nvfp4_ref` (commit `619f99fef`)
produces strictly better N4_0 *weight* quantization than the single-code baseline.

Concretely: Qwen3-1.7B-Base quantized to N4_0 **with** the search will have lower
KL-divergence from its f16 reference (and lower perplexity) than the same model quantized
**without** it.

- Prior signal: `test-quantize-fns -v` n4_0 dot-product error dropped 0.01977 -> 0.00242 (~8x)
  and absolute error 0.00234 -> 0.00203 when the search was added. We expect this to propagate
  to a clear end-to-end gain.
- Bound: the search only refines each 16-element sub-block's UE4M3 scale; it cannot restore
  N4_0's dropped per-tensor scale. So it should *narrow* the f16->n4_0 gap, not close it.
- Null hypothesis: searched KLD/PPL ~= unsearched (the synthetic gain does not survive on real
  weights).

## Prior art

- https://github.com/ggml-org/llama.cpp/pull/22858 ("llama-quantizer has no code to make correct NVFP4 ggufs")
- https://github.com/ggml-org/llama.cpp/pull/23046 (NvFP4 quantized LM head support)
- https://github.com/vllm-project/vllm/pull/42124 (LM head quantization support for ModelOpt)
- https://github.com/ggml-org/llama.cpp/pull/22196 (Blackwell native NVFP4 support)
- https://github.com/ggml-org/llama.cpp/pull/21074 (generic NVFP4 MMQ kernel)
- https://github.com/ggml-org/llama.cpp/pull/20506 (Qwen3.5/Qwen3.5MoE tensors for NVFP4)
- https://github.com/ggml-org/llama.cpp/pull/22897 (NVFP4 scale tensors)
- https://github.com/ggml-org/llama.cpp/pull/23484 (scaled GEMMs for more robust NVFP4 support)

### Pior art review

The upstream NVFP4 work splits cleanly into **kernels** (consume scales), a **consumer** (applies
the per-tensor scale in the graph), and **producers** (emit the scale tensors). Read in that frame:

Kernels (consume the per-16 UE4M3 block scales; both assume the per-tensor global scale is `1`):
- https://github.com/ggml-org/llama.cpp/pull/22196 (Blackwell native NVFP4 / FP4 OMMA). The
  `scale_vec::4X` MMA reads the four UE4M3 block scales via `get_int_b4` into one packed register
  with `{0,0}` PTX selectors; the lane map (`tidx_A = tx/4 + (tx%2)*8`) selects which *row's* scale
  register a lane supplies, not which byte - so it is **structurally magnitude-independent** (a code
  reading, not a proof). Validated only on `test-backend-ops` (near-uniform synthetic) and
  self-quantized (subnormal) ggufs; the real normal-range/high-variance regime is untested in-PR.
  NVIDIA's per-tensor F32 scale is deliberately deferred to a separate `GGML_OP_MUL` (discussion
  #22042: `quantize/dequantize_row_nvfp4` are "incorrect when F32 != 1.0").
- https://github.com/ggml-org/llama.cpp/pull/21074 (generic NVFP4 MMQ, non-Blackwell dp4a). Each
  lane decodes its own block's four UE4M3 scales independently (`ggml_cuda_ue4m3_to_fp32` ->
  `x_df`), no warp cooperation - **immune** to any lane-permutation scale bug. Same per-block decode
  as `dequantize_block_nvfp4`. Validated for speed (`llama-bench`) only.

Consumer (the per-tensor scale path - present on master *and* on this branch):
- `build_lora_mm(w, cur, w_s)` -> `res = ggml_mul(ctx0, res, w_s)` (`llama-graph.cpp:1002`). The
  weight `.scale` ("NVFP4 scale2") is a post-matmul per-tensor multiply that rides on top of *any*
  mul_mat kernel (native FP4 or dequant). This is where NVIDIA's dropped per-tensor F32 global
  re-enters - as a sibling tensor + graph epilogue, **no new ggml type, block layout unchanged**.
  The activation `.input_scale` is loaded and saved but **consumed nowhere** (dead across the whole
  `src/` tree; `build_lora_mm` has no input-scale parameter) - scaffolding for a future W4A4 path.

Producers (emit `.scale` / `.input_scale`; two independent ones):
- https://github.com/ggml-org/llama.cpp/pull/22897 (`llama-quantize`, OPEN). Adds the missing
  `LLAMA_FTYPE_MOSTLY_NVFP4` default-type mapping (was UB) and emits per-tensor F32 `.scale` =
  `sum(f32 * dequant) / sum(dequant^2)` - the **LS-optimal global bias correction** given the
  *unchanged subnormal* block scales (their 0.5B run: ~0.96-1.04). It does **not** touch the
  block-scale encoding, so it is orthogonal to (and far smaller than) the subnormal rescue measured
  below. Writes `.input_scale` = the same value (a placeholder; inert). Experts (`.experts.`)
  excluded. Supersedes the closed #22858 ("llama-quantizer has no code to make correct NVFP4 ggufs").
- https://github.com/ggml-org/llama.cpp/pull/20506 + #20505 (Qwen3.5/Qwen3.5-MoE). #20506 wires the
  `*_s` weight scales into the graph; the sibling #20505 is the converter that imports a
  pre-quantized ModelOpt/compressed-tensors NVFP4 checkpoint **verbatim** - block UE4M3 scales
  bit-copied (`& 0x7F`), `weight_scale_2` -> `.scale` (raw F32, applied via `ggml_mul`),
  `input_scale` -> `.input_scale` (dead). Self-quantizer bypassed (`raw_dtype=NVFP4`). These are the
  only ggufs that carry genuine normal-range block scales + a real global scale. **Not on this
  branch** (`feat/nvfp4-gguf`): the loader half is here, the converter half is not.
- https://github.com/ggml-org/llama.cpp/pull/23046 (NVFP4 LM head). Extends `.scale` to the
  top-level `output` tensor (outside the per-layer loop), wires `output_s` into the final-logits
  `build_lora_mm`, and bans NVFP4 on a tied `output == tok_embd` head. `.input_scale` still dead.
- https://github.com/vllm-project/vllm/pull/42124 (vLLM LM-head quant for ModelOpt; reference).

## Setup

- Model: Qwen/Qwen3-1.7B-Base (dense), converted to f16 GGUF as the reference.
- Corpus: `wikitext-2-raw/wiki.test.raw`, ctx 512, `-b 4096 -ub 2048`.
- Base logits: `kld-base-qwen3.dat` (from the f16 model).
- A/B: `n4_0-new` = HEAD (search), `n4_0-old` = HEAD~1 (single-code), `q4_0` = same-bpw context.
- Quantizer change is `quantize_row_nvfp4_ref` only; `llama-quantize` N4_0 selectability added
  separately and present in both arms. CPU-side quantization; imatrix not used (constant across A/B).

## Honest framing

- This measures ggml's single-level N4_0 (old vs new search), not NVIDIA's two-level PTQ NVFP4.
- `quantize_row_nvfp4_ref` ignores the imatrix.

## Results

Qwen3-1.7B-Base, full `wiki.test.raw`, ctx 512, b4096/ub2048. KLD/PPL vs the f16 base
(PPL(f16) ~= 9.87). All three quant arms are 4.87 bpw (identical GGUF size, 999.90 MiB):
n4_0 tensors + Q6_K output + Q8_0 embd, same mix in every arm.

| quant                  | Mean PPL(Q)        | PPL ratio | Mean KLD            | Same top-p |
|------------------------|--------------------|-----------|---------------------|------------|
| n4_0-old (no search)  | 11.790 +/- 0.086   | 1.19411   | 0.19044 +/- 0.00081 | 78.49%     |
| n4_0-new (MSE search) | 11.583 +/- 0.082   | 1.17312   | 0.17393 +/- 0.00069 | 78.62%     |
| q4_0 (context)         | 11.387 +/- 0.081   | 1.15325   | 0.16130 +/- 0.00046 | 79.20%     |

**Hypothesis CONFIRMED.** The MSE scale search lowers N4_0 weight-quant divergence:
Mean KLD 0.19044 -> 0.17393 (**-8.7%**), PPL penalty +19.4% -> +17.3%. The gap is ~20 sigma
on the error bars - robustly real, not noise.

**But the gain is modest, and bounded as predicted.** The ~8x synthetic dot-error improvement
(`test-quantize-fns`, uniform random data) does NOT translate to ~8x end-to-end: real weights
see ~9% lower KLD. The search refines each sub-block's UE4M3 scale but cannot restore N4_0's
dropped per-tensor scale.

**n4_0 still trails q4_0 at equal footprint** (searched-n4_0 0.17393 vs q4_0 0.16130, ~8%
worse KLD). So the search makes n4_0 *better than before*, not *better than q4_0* - consistent
with the KV-cache finding: ggml's single-level n4_0 tracks q4_0 and never beats it.

Same-top-p (top-1 token agreement vs the f16 base) tells the same story at coarser resolution:
q4_0 79.20% vs n4_0 78.62%, a **~0.58 pp** gap in q4_0's favor (the search narrows it from
~0.71 pp at no-search n4_0 = 78.49%, but does not overtake). Same-top-p only counts the argmax;
the full-distribution shift shows up larger in KLD (the ~8% above).

**Takeaway.** The MSE search is a worthwhile, free improvement to the *existing* N4_0
reference quantizer (~9% lower KLD for any n4_0 weight conversion or KV write, and it removed
CPU/CUDA encoding drift). It does not make n4_0 the preferred 4-bit format. Caveats: single
small model; no imatrix (n4_0 ignores it - q4_0/q4_K with imatrix would widen their lead).

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

Plumbed the imatrix into n4_0: `quantize_nvfp4` now branches on `quant_weights`, and
`quantize_row_nvfp4_impl` weights the +/-2 UE4M3 scale search by `qw*sqrt(sigma2+x^2)` (same
form as `quantize_row_q4_0_impl`). q4_0 re-quantized with the same imatrix (overwrites the
no-imatrix q4_0).

|       | no-imatrix KLD | imatrix KLD | imatrix gain |
|-------|----------------|-------------|--------------|
| q4_0  | 0.16130        | 0.10145     | -37.1%       |
| n4_0 | 0.17393        | 0.14431     | -17.0%       |

The n4_0 imatrix plumbing **works** (-17%, best n4_0 result; a real keepable improvement,
unlike +/-3 and x^2). But q4_0 exploits the imatrix >2x as well, so the **gap widens**:
n4_0-imat is 42% worse than q4_0-imat (0.1443 vs 0.1015), up from 8% without imatrix.

Why n4_0 can't ride it as far (the structural ceiling, sharpened): q4_0's `make_qx_quants`
lets the imatrix steer a *continuous* fp16 scale (fine +/-9x0.1 grid) AND the per-element level
assignment (weighted least-squares). n4_0's path can only let the imatrix pick among ~5
*discrete* UE4M3 rungs, with per-element E2M1 codes still nearest (fixed non-uniform levels -
no room to spend importance on a value). Coarse discrete scale throttles the benefit.

## Overall conclusion

On quality, q4_0 beats n4_0 at every level and the imatrix widens the lead - the E2M1 +
per-16 FP8-scale format is the ceiling, not the quantizer. n4_0's case is purely **prefill
throughput**: native FP4 MMA on Blackwell gives +~19% pp on the 7B (-2% decode). So n4_0 is a
speed-for-quality trade, not a quality win. The two keepable contributions, independent of that
verdict: (1) the +/-2 MSE scale search (~9% better n4_0 quant, fixed CPU/CUDA drift), (2) the
imatrix plumbing (-17%). Both improve the existing type; neither makes n4_0 preferable to q4_0
on quality.

## Performance (RTX 5090, native FP4 vs q4_0)

`llama-bench`, `-ngl 99 -b 4096 -ub 2048 -r 20`. Each pair is the same f16 source quantized to
n4_0 (committed +/-2 search) and q4_0; identical GGUF size in each pair (so equal bytes/bandwidth).
On Blackwell the n4_0 weight matmul takes the native FP4 MMA path (`mmq.cu:125` `use_native_fp4`);
q4_0 uses integer DP4A MMQ. Decode (`tg`, batch 1) is a bandwidth-bound GEMV via MMVQ - no tensor
cores on either side.

Qwen3-1.7B (t/s):

| test    | q4_0            | n4_0           | delta  |
|---------|-----------------|-----------------|--------|
| pp4096  | 23967 +/- 258   | 24649 +/- 333   | +2.8%  |
| pp8192  | 17713 +/- 112   | 18052 +/- 263   | +1.9%  |
| pp16384 | 11175 +/- 355   | 11330 +/- 314   | +1.4%  |
| tg128   | 588.0 +/- 16    | 558.3 +/- 13    | -5.0%  |

Qwen2.5-7B (t/s):

| test    | q4_0            | n4_0           | delta   |
|---------|-----------------|-----------------|---------|
| pp4096  | 10944 +/- 89    | 13065 +/- 16    | +19.4%  |
| pp8192  | 8518 +/- 15     | 9738 +/- 15     | +14.3%  |
| pp16384 | 5833 +/- 23     | 6394 +/- 12     | +9.6%   |
| tg128   | 281.2 +/- 0.9   | 275.4 +/- 0.7   | -2.1%   |

Qwen3.6-27B (dense Qwen3.5, t/s, **-b 8192 -ub 4096** "full blast", + Q3_K arm):

| test    | q4_0 (4.5bpw)   | n4_0 (4.5bpw)  | Q3_K (3.4bpw)   | n4_0 vs q4_0 | n4_0 vs Q3_K |
|---------|-----------------|-----------------|-----------------|---------------|---------------|
| pp4096  | 3635.6 +/- 25   | 4804.5 +/- 31   | 3048.7 +/- 9    | +32.2%        | +57.6%        |
| pp8192  | 3427.0 +/- 38   | 4564.9 +/- 27   | 2945.0 +/- 10   | +33.2%        | +55.0%        |
| pp16384 | 3051.8 +/- 10   | 3948.5 +/- 5    | 2644.7 +/- 4    | +29.4%        | +49.3%        |
| tg128   | 84.42 +/- 0.13  | 84.08 +/- 0.12  | 85.65 +/- 0.09  | -0.4%         | -1.8%         |

At 27B (full blast) the prefill win **strengthens and flattens**: ~+30% over q4_0 across all
prompt lengths (the FFN body so dominates a 27B that attention dilution barely registers). The
**decode tax vanishes with scale**: -5.0% (1.7B) -> -2.1% (7B) -> -0.4% (27B), as the per-16
FP8/LUT unpack amortizes. And n4_0 **beats the smaller Q3_K on prefill by +49-58%** despite
Q3_K's fewer bits: prefill is compute-bound, Q3_K runs DP4A K-quant (no tensor cores), so 3.4
bpw is *slower* there; Q3_K only edges decode (+1.8%, bandwidth-bound). Speed landscape: prefill
n4_0 >> q4_0 > Q3_K; decode all within ~2%. Power pinned ~572/575 W during prefill, so the +30%
prefill ~= ~30% fewer joules/prompt vs q4_0 (and below Q3_K's energy too).

**Prefill: n4_0 wins, and the win scales with model size** (1.7B ~1-3%, 7B ~10-19%, 27B ~+30%). That
scaling IS the evidence the FP4 cores are doing the work: native FP4 MMA needs large GEMMs to
flex, and a 1.7B's FFN matmuls are too small. The taper with prompt length (+19 -> +14 -> +10%
at 7B) is attention dilution - O(n^2) f16 attention is identical for both arms and grows as a
share of total time, shrinking the weight-matmul fraction the format difference rides on.

**Decode: n4_0 loses** (~2% at 7B, ~5% at 1.7B). Same bytes -> same bandwidth; the loss is the
extra per-byte unpack in the GEMV - non-uniform E2M1 LUT lookup (`get_int_from_table_16`) plus a
per-16 UE4M3/FP8 scale decode - that q4_0 avoids (uniform codes straight into DP4A, one fp16
scale per 32). The FP4 MMA that wins prefill is absent on this path, so the format's richness is
pure overhead here. It's the exact mirror of the prefill win.

**Energy:** at ~equal power, +16% prefill t/s (pp4096, 7B) ~= ~16% fewer joules for the prefill
phase; decode ~neutral. So for prefill-of-long-prompts, n4_0 is faster and cooler. (Power not
yet sampled; this is the speed->energy floor, not a measured J/token.)

**This gain is NOT free - it is a trade, not a Pareto win.** Pair it with the quality result
above: n4_0 carries ~8% worse KLD than q4_0 at equal footprint. So n4_0 on Blackwell buys
**+10-19% prefill throughput at the cost of ~8% KLD / ~0.58 pp top-1**, same size. Worth it if
prefill-/energy-bound and tolerant of a small quality dip; q4_0 still wins quality-per-byte. The
imatrix asymmetry tilts the trade further toward q4 (it would widen q4's quality lead while
n4_0's speed edge persists).

**Native FP4 confirmed three independent ways (no ambiguity):**
- *Compiled* - `cuobjdump --dump-sass libggml-cuda.so` carries `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`
  (= NVFP4: m16n8**k64** tile, E2M1 4-bit operands, UE4M3 block scale, scale_vec::4X; the `.E8`
  sibling is MXFP4). That is `mma.sync...kind::mxf4nvf4` lowered to SASS.
- *Dispatched* - `nsys` on an n4_0 prefill shows `quantize_mmq_nvfp4` (the FP4 activation packer,
  unique to the `use_native_fp4` branch) + `mul_mat_q<(ggml_type)40,...>` (N4_0) as the top kernels.
- *Executed* - `ncu` profiles `mul_mat_q<40>` on CC 12.0 (RTX 5090).

**Mechanism of the prefill win (from the SASS):** q4_0's MMQ runs `IMMA.16832.S8.S8` (int8,
**k=32**); n4_0 runs `OMMA.16864...` (FP4, **k=64**) - double the K-depth per tensor-core
instruction at the same issue slots. That k32->k64 doubling is the architectural source of the
+10-19%, not a vague "FP4 is faster". (ncu also showed the kernel only ~17% SM-busy at pp256 -
the FP4 cores idle at small sizes, which is why the win scales with size/batch -> the batched
avenue below.)

Caveats: clocks unlocked, but 7B sigmas are ~0.1-0.8% (runs long enough for stable boost).
Same-model quality is below.

## 7B quality (Qwen2.5-7B) + the same-model verdict

Weight-quant KLD/PPL vs the qwen2.5-7b f16 base (full `wiki.test.raw`, ctx 512). n4_0 and
q4_0-fresh are the same ~4.13 GiB; q4_0-imat is +0.3% (imatrix nudges a couple tensor types).

| quant            | PPL ratio | Mean KLD            | Same top-p |
|------------------|-----------|---------------------|------------|
| n4_0 (search)   | +7.7%     | 0.10524             | 85.21%     |
| n4_0 + imatrix  | +7.1%     | 0.09235             | 86.33%     |
| q4_0 (no imat)   | +4.1%     | 0.05779             | 88.83%     |
| q4_0 + imatrix   | +1.8%     | 0.04649             | 89.92%     |

- **n4_0 vs plain q4_0: ~82% worse KLD** (0.105 vs 0.058), ~1.9x the PPL penalty, -3.6 pp
  top-1. At 1.7B (Qwen3) the gap was only ~8%; at 7B (Qwen2.5) it blows open.
- **imatrix helps both, unevenly.** q4_0 halves its PPL penalty (+4.1% -> +1.8%, -20% KLD);
  n4_0 - once plumbed (`quantize_row_nvfp4_impl`) - gains -12% KLD (0.1052 -> 0.0924). n4_0
  *can* use the imatrix now; it just rides it less far (single-scale ceiling, see below).
- **n4_0+imatrix vs the realistic q4_0+imatrix: ~2x worse KLD** (0.0924 vs 0.0465), ~3.9x the
  PPL penalty (+7.1% vs +1.8%), -3.6 pp top-1.

**Same-model speed-vs-quality trade:** n4_0 buys +10-19% prefill at ~2x worse quantization
quality than q4_0 (imatrix on both: 0.0924 vs 0.0465). Whether that's worth it is
workload-dependent, not a wash: prefill-/throughput-/energy-bound work on consumer Blackwell
(the whole sm_120 line, 5050..5090 + laptop parts) can reasonably take the trade; memory-bound
or quality-per-byte favors q4_0/IQ4_XS. A real trade, not a free win and not a write-off.

**Why 8% (1.7B) -> 82% (7B):** n4_0's single-level floor. q4_0 rides its fp16-per-32 scale down
to 0.058 on the redundant 7B; n4_0, missing the per-tensor scale, stalls at 0.105 and cannot
follow. On the harder 1.7B both sit near a ~0.16-0.17 floor and look close. n4_0 looks
competitive exactly where 4-bit is hard, and falls behind exactly where 4-bit is easy. (Caveat:
1.7B=Qwen3, 7B=Qwen2.5 - same family that showed K-outliers in the KV study - so model and scale
are confounded; the 7B result itself is clean.)

## Quality-per-bit ladder (Qwen2.5-7B, same f16 base, no imatrix unless noted)

Where n4_0 actually sits among 4-bit-class formats. Sorted by KLD (best first):

| format          | size      | Mean KLD | PPL ratio | Same top-p |
|-----------------|-----------|----------|-----------|------------|
| IQ4_XS          | 3.96 GiB  | 0.04071  | +3.0%     | 90.51%     |
| q4_0 + imatrix  | 4.14 GiB  | 0.04649  | +1.8%     | 89.92%     |
| q4_0            | 4.13 GiB  | 0.05779  | +4.1%     | 88.83%     |
| **n4_0 + imat**| 4.13 GiB  | 0.09235  | +7.1%     | 86.33%     |
| n4_0 (search)  | 4.13 GiB  | 0.10524  | +7.7%     | 85.21%     |
| Q3_K_M          | 3.55 GiB  | 0.10925  | +8.6%     | 85.08%     |
| Q3_K_S          | 3.25 GiB  | 0.19372  | +12.2%    | 79.57%     |

**n4_0 sits at the q3/q4 boundary in quality; the imatrix lifts it but cannot reach real 4-bit.**
- Without imatrix it merely matches Q3_K_M (0.1052 vs 0.1092 KLD) at 0.58 GiB more size. With the
  imatrix it **clears Q3_K_M** (0.0924 vs 0.1092, 86.33% vs 85.08% top-1) - so it's no longer
  q3-tier, but it's still not q4-tier.
- It remains dominated on quality-per-byte by the real 4-bit formats: q4_0 (0.058) and q4_0+imatrix
  (0.046) at the same size, and IQ4_XS (0.041) which is *smaller* (3.96 GiB) **and** ~2x better KLD.

So on quality-per-byte alone, q4_0/IQ4_XS win and n4_0 is the wrong tool for the memory-bound
"fit a bigger model" case. n4_0's non-dominated property is **Blackwell native FP4 prefill
throughput** (+10-19%, see Performance) - available across the whole consumer sm_120 line, not
just halo cards. That makes it a genuine trade for prefill-/throughput-/energy-bound workloads:
spend ~2x quantization quality (vs q4_0+imatrix) to buy double-digit prefill on commodity
Blackwell. Worth it or not depends on the workload; it is not a blanket write-off.

## Why the imatrix helps n4_0 less: the single-scale ceiling

The imatrix's only lever is the **scale choice** - in *both* formats. For a fixed scale `d` the
weighted block error `sum_j w_j (x_j - c_j*d)^2` separates per element, so each code
`c_j = argmin_c w_j (x_j - c*d)^2` = nearest code, independent of `w_j` (a non-negative constant
just scales that term). Importance weights only matter where the error is *summed across j* -
i.e. picking `d`. q4_0's per-element levels are nearest-given-scale too; the imatrix does not
touch per-element rounding in either.

So the binding constraint is purely the **scale's resolution**:
- q4_0: `d` is a **continuous fp16** value (LS-fit via `sumlx/suml2`, finely searched). The
  importance-weighted optimum is a real number and a continuous knob lands on it.
- n4_0: `d` is a **UE4M3** value; the +/-2 search reaches ~5 of them. The weighted optimum is
  *rounded to the nearest reachable rung*. The imatrix computes a better target; n4_0 cannot
  represent it. Hence -17% (n4_0) vs -37% (q4_0).

**Unifying point:** that single coarse scale is exactly ggml dropping real-NVFP4's second-level
**per-tensor FP32 scale** (true NVFP4 effective scale = `FP32_tensor * UE4M3_subblock`; ggml
kept only the UE4M3 micro-scale). This is the *same* omission that makes the block writable
incrementally - i.e. KV-viable. One design choice, three consequences: KV-viability (+), capped
weight quality vs q4_0 (-), and a starved imatrix lever (-).

**Actionable, and it splits weights from KV:**
- **Weights are static** -> two-level scaling is free (it is how real NVFP4 PTQ checkpoints
  store weights). Adding a per-tensor FP32 scale would hand the imatrix a continuous knob and
  very likely recover most of the gap to q4_0 - the real path to "n4_0 weights competitive
  *and* +19% prefill". It is a format change (extra per-tensor scale, the reserved **N4_K**), not a quantizer tweak.
- **KV is block-local** -> the cache is written per token, so a per-tensor scale is impossible;
  the coarse ceiling is inherent and the KV verdict (tracks q4_0, never better) is permanent.

One-liner: the imatrix does not generalize *poorly* for n4_0 - it generalizes exactly as far
as the format's one coarse degree of freedom allows, and that freedom was traded for
KV-viability. Restore the second scale level (weights only) and the imatrix story changes.

(Generalization caveats for the magnitudes: the imatrix was calibrated on wiki.train and
evaluated on wiki.test - same domain, so -17%/-37% are optimistic in-domain bounds; a diverse
calibration set trades some in-domain gain for robustness. The *relative* claim - n4_0 gains
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

## No-subnormal ceiling + the W4A4 activation confound (Qwen3.6-27B KLD)

Wrapped UE4M3 with a single global power-of-2 shift (N=6: encode `*2^N`, decode `*2^-N`) so block
scales land in the 3-bit-mantissa normal grid instead of the subnormal floor; requantized
Qwen3.6-27B; KLD vs the f16 base (200 chunks, wikitext-2, c=512). The native FP4 OMMA decodes UE4M3
in hardware, so no-subnormal codes can only be read on the **dequant path** - which, crucially, also
keeps activations at f16 (W4A16) instead of fp4 (W4A4). An N=0 dequant control disentangles the two:

| config                       | activations | weight scales | Mean KLD            |
|------------------------------|-------------|---------------|---------------------|
| n4_0 native (deployed)      | fp4  (W4A4) | subnormal     | 0.07907 +/- 0.00175 |
| n4_0 dequant, N=0 (control) | f16  (W4A16)| subnormal     | 0.05003 +/- 0.00150 |
| n4_0 dequant, no-subnormal  | f16  (W4A16)| normal        | 0.04028 +/- 0.00140 |
| q4_0 native (deployed)       | int8 (W4A8) | fp16          | 0.04852 +/- 0.00149 |
| q4_0 dequant (W4A16)         | f16  (W4A16)| fp16          | 0.04815 +/- 0.00149 |

Two independent effects, previously conflated:
- **Subnormal rescue (weight format), W4A16-isolated: 0.0500 -> 0.0403, -19%.** Real and free - the
  `amax/6`+`*0.5` convention parks ~100% of weight scales in the subnormal grid; a global exponent
  shift recovers 3-bit-mantissa precision. (An earlier "-49%" reading compared W4A16-nosubn against
  W4A4-native, which also flipped activations to f16 - **retracted**; the clean weight-only delta
  is -19%.)
- **Activation precision (path), W4A4 vs W4A16: +0.029 KLD.** Native FP4 quantizes *activations* to
  fp4 (E2M1, 8 levels) too; this is the single largest error in native n4_0 - larger than the
  weight quantization itself - and the weight recalibration does not touch it. q4_0's MMQ uses
  **int8** activations and q4_0 is ~insensitive to it (W4A8 0.04852 ~= W4A16 0.04815 - int8 acts are
  near-lossless), whereas n4_0's fp4 acts cost +0.029.

Verdict (now apples-to-apples):
- **As a weight format (W4A16, f16 acts), no-subnormal n4_0 BEATS q4_0: 0.0403 vs 0.0482 (-16%).**
  The subnormal fix flips n4_0 from *behind* q4_0 (standard 0.0500) to *ahead* - 4-bit E2M1 + a
  precise per-16 UE4M3 scale is a better weight code than q4_0's uniform int4 + fp16-per-32, once
  the scale isn't crippled. No per-tensor storage, no new type; the fix is a global exponent shift.
- **In the deployed native FP4 path (the speed win), n4_0 stays behind q4_0** because it is
  **activation-limited**: W4A4 fp4 activations cost ~0.029 KLD that q4_0's int8 avoids. Applying the
  weight fix natively (a constant `2^-N` epilogue on the MMA accumulator) cuts only the weight
  component (~-0.01 -> est. ~0.069 W4A4), still above q4_0 native 0.0485. And native FP4 *requires*
  W4A4 (the OMMA is fp4 x fp4) - you cannot pair fp4 weights with int8 acts on the FP4 cores - so the
  activation penalty is intrinsic to using the speed path.

This refines the doc's "single coarse scale" thesis: n4_0 trailed q4_0 for *two* reasons -
subnormal weight scales (fixable, -19%, and once fixed the weight format wins) and W4A4-vs-W4A8
activations (intrinsic to native FP4, the dominant deployed error) - not the per-tensor scale's
resolution. The actionable wins: (1) recalibrate the UE4M3 scale placement (free, helps every n4_0
model, weight-only and KV); (2) the FP4-speed-vs-quality trade is really a W4A4-activation trade.

## Native FP4 rejects rebiased scales (the speed/quality fork)

Tried to carry the no-subnormal weight fix onto the native FP4 path: re-encode scales into the
normal grid (lift x2^6), then compensate at the OMMA write-back (mmq.cuh:3257, `* 2^-N`, gated to
N4_0). It **breaks non-uniformly** (Qwen3.6-27B, lifted-scale gguf):

| path                | activations | epilogue | Mean KLD | Same top-p |
|---------------------|-------------|----------|----------|------------|
| dequant (cuBLAS f16)| W4A16       | n/a      | 0.0403   | 92.49%     |
| native FP4 OMMA     | W4A4        | none     | 12.64    | 0.000%     |
| native FP4 OMMA     | W4A4        | x 2^-6   | 3.87     | 26.05%     |

The *same* lifted gguf is correct on the dequant path (0.0403, scales decoded by the same hardware
`__nv_fp8_e4m3`) but collapses on the OMMA - top-p ->0. A uniform `2^-N` cannot fix it: a global
factor preserves the argmax, so top-p 0%/26% means **per-block** corruption, not a scale-factor
error. The OMMA's `block_scale` instruction consumes rebiased weight scales wrongly.

**Likely a ggml kernel bug, not a hardware limit.** Real NVFP4 (TensorRT-LLM) runs block scales up
to E4M3's max (448) on this same OMMA; our lifted scales peak at ~15. ggml's native-FP4 N4_0 path
has only ever seen the scales ggml emits - 99.94% subnormal - so the normal-range regime is untested
(suspect `load_tiles_nvfp4` scale packing or the `scale_vec::4X` operand setup, mma.cuh:1145).

**The fork this creates for n4_0/N4_K:**
- The weight-quality fix (normal-grid scales, 0.0403, beats q4_0) works **only on W4A16** today -
  which forgoes the native FP4 speed.
- On the native FP4 speed path, n4_0 is stuck near 0.0791 (subnormal scales) and is *also*
  W4A4-activation-limited - doubly behind q4_0.
- So n4_0 currently splits: **speed (native, ~0.079, < q4_0) XOR quality (W4A16, 0.0403, > q4_0)**.
  Unlocking N4_K on the speed path requires first fixing the native kernel's normal-range scale
  handling (or adopting real NVFP4's normalized two-level layout that the OMMA expects).

## Future avenue: batched / concurrent-prefill serving

All perf numbers above are single-stream `pp` (+10-19% n4_0 on 7B). The strongest unmeasured
case for n4_0 is **batched serving** - many parallel prefills, long-context RAG, agentic
workloads - which is *more* compute-bound, so the native FP4 advantage should grow beyond the
single-stream figure. This is n4_0's best-case workload and the natural home for the
throughput-bound segment. Native FP4 is the entire consumer Blackwell line (sm_120: 5050..5090),
not a 5090 halo feature, so the segment is mass-market; the +10-19% ratio is a 5090 number that
should roughly transfer down the line but is unmeasured on smaller cards. Worth a llama-bench
(multiple parallel sequences) or server throughput run before a final line on n4_0's usefulness.
This is orthogonal to the weight-quality avenue above: batched prefill widens n4_0's *speed*
edge; the two-level-scale restoration would fix its *quality* deficit - independent levers.

## Producer/consumer split + corrected kernel-bug status (2026-05-22)

Reviewing the upstream PRs (kernels #22196/#21074, consumer `build_lora_mm`, producers #22897 and
#20505/#20506/#23046, see Prior art) reframes the "second scale level" question and walks back the
"likely a ggml kernel bug" conclusion of the *Native FP4 rejects rebiased scales* section above.

**The per-tensor scale already has a home, and it is not the kernel.** NVIDIA's dropped F32 global
scale re-enters as a sibling `.scale` tensor multiplied onto the matmul output in the graph
(`build_lora_mm` -> `ggml_mul`), riding on top of *either* the native FP4 OMMA or the dequant
kernel. The kernels deliberately assume `s_global == 1`; the maintainers split responsibility so the
block layout (and the FP4 OMMA) never has to change. So the reserved **N4_K** needs **no new ggml
type** - it is N4_0's block plus a per-tensor F32 `.scale`, which the loader on this branch already
applies. The activation companion `.input_scale` is emitted by both producers but consumed nowhere
(`build_lora_mm` has no input-scale arg) - dead scaffolding for a future W4A4 path.

**Two distinct levers, previously conflated under "second scale level":**
- **(a) LS bias correction** - #22897's `.scale = sum(f32*dq)/sum(dq^2)`, a single per-tensor scalar
  that removes the residual *multiplicative* bias given the *unchanged subnormal* block scales
  (~0.96-1.04). It cannot fix per-block or per-element error, only the mean gain - expect a small
  KLD move. Works on native (it is a graph epilogue).
- **(b) Subnormal rescue** - re-encoding the block UE4M3 scales into the normal 3-bit-mantissa grid
  (the -19% W4A16 result above). This changes the *block-scale encoding*, which #22897 does **not**
  do. This is the dominant weight-format gap. Canonical NVFP4 gets (b) for free: a per-tensor
  `s_global = global_amax/(448*6)` normalizes block scales into the normal range *and* is stored as
  `.scale` - one factor delivers both the rescue and the magnitude compensation.

**Kernel-bug status: contested, not confirmed.** The native scale operand reads the four block
scales into one register with row-selecting lanes (#22196) and the generic path decodes each scale
independently (#21074) - both arguments say normal-range, high-variance scales should *not* be
corrupted by the kernel. Yet the lifted-scale `nosubn` gguf gave top-p 26% on native (above). The
prime suspect is now the *measurement vehicle*, not the kernel: that run carried the magnitude
compensation as a `* 2^-N` patch in the MMQ **write-back**, which only covers the matmuls that hit
the instrumented epilogue (MMVQ decode, stream-k fixup, MoE/`mul_mat_id`, and the router were never
proven covered) and compounds any miss across 65 layers. The `ggml_mul` consumer path has none of
that: it scales *every* matmul output uniformly, in the graph, independent of which CUDA kernel ran.

**Clean decisive test (now unblocked by the #22897 producer mechanism):** quantize with block scales
re-encoded into the normal grid **and** a compensating per-tensor `.scale` written as a sibling
tensor (extend #22897's emission; set `.scale = 2^-N * LS_correction`), then run on the native FP4
OMMA on a **stock** binary - the compensation arrives via `ggml_mul`, the write-back hack is gone.
- SOUND + KLD ~ 0.069 (0.040 weight-rescued + 0.029 W4A4 activation) -> no kernel bug; the earlier
  26% was the write-back hack's path-coverage gap, and the speed/quality fork stands as a W4A4
  *activation* trade (the weight rescue helps only W4A16).
- GARBAGE -> a genuine native normal-range scale issue survives the clean epilogue; pin it with a
  deterministic high-variance `test-backend-ops` case before any kernel change.
Vehicle note: Qwen3.6-27B is `qwen35` **dense** (block_count 65), so `build_lora_mm` covers all its
weights - no `build_lora_mm_id`/expert-scale wrinkle (and #22897 excludes experts anyway).

**Activation side is not a scale-placement problem.** The native activation quantizer
(`quantize.cu`) computes its own per-16 UE4M3 activation scale dynamically (`amax_raw/6`, +/-2 MSE)
and ignores `.input_scale`. Activation amax is O(0.1-1) -> those block scales already land in normal
range, so the +0.029 W4A4 penalty is intrinsic E2M1 4-bit resolution, **not** a subnormal-placement
loss a global scale could rescue. Wiring `.input_scale` would not recover it.

**Smoke gate.** `nvfp4-smoke.sh` now greedy-decodes a few tokens from a repeated `passed\n` prompt
(`llama-completion --temp 0 --top-k 1`) and checks the model echoes it - a fast coherence gate, not
a quality run. Stock build: `q4_0` SOUND, `nvfp4` SOUND, `nvfp4-nosubn` GARBAGE (expected - the
lifted bytes only reconstruct on the matching divide-by-64 build; this is the artifact the clean
test above replaces).

## Decisive test resolved: native kernel works, decomposed by a W4A16 dequant control (2026-05-22)

Ran the clean test the section above proposed. Quantized Qwen3.6-27B with the subnormal lift in the
quantizer (`quantize_row_nvfp4_ref`/`_impl`: encode arg `*64`, decode `/64`; global UE4M3 codec left
stock) + the #22897 producer port, so the stored byte reads 64x large on every path and the
per-tensor `.scale = sum(f32*dq)/sum(dq^2)` LS-derives `2^-6 * bias` (~0.01564) automatically,
applied via `build_lora_mm`'s `ggml_mul`. Built `qwen3.6-27b-nvfp4-canon2.gguf`.

**It runs SOUND on the native FP4 OMMA, coherent generation - the "kernel bug" is refuted.** The
native scale operand is magnitude-independent after all (PR #22196/#21074 review: four UE4M3 scales
in one register, scale_vec::4X `{0,0}` selectors, `tidx_A` picks the row not the byte). The earlier
26%/top-p-0 came from doing the compensation in the MMQ *write-back* (missed MMVQ/stream-k/router and
compounded across 65 layers); the graph-level `ggml_mul` epilogue covers every matmul uniformly,
kernel-agnostic.

**Coverage gap (debugged via a load failure).** The `.scale` epilogue only covers matmul weights.
Two NVFP4 paths apply no scale and must stay unlifted (`--token-embedding-type q8_0
--tensor-type eh_proj=q8_0`): `token_embd` (read by `get_rows` -> 64x embeddings) and `nextn.eh_proj`
(`build_lora_mm` with no `_s` arg; #22897 also over-emits its scale, two unconsumed tensors ->
`done_getting_tensors: expected 1874, got 1872`). This is the same gap the tied-`output==tok_embd`
assert guards, and it applies equally to NVIDIA's canonical per-tensor `s_global`.

**The dequant control (W4A16).** To separate weight-quant quality from the W4A4 activation penalty,
force NVFP4 off MMQ onto the cuBLAS dequant path: env gate `GGML_NVFP4_FORCE_DEQUANT=1` ->
`ggml_cuda_should_use_mmq` returns false for NVFP4 -> weights dequantized to f16, f16 GEMM, f16 acts.
`.scale` still applies (it is graph-level). Same gguf, same 200-chunk KLD vs the f16 base:

| config                             | acts        | weight scales | Mean KLD            | PPL ratio | top-p   |
|------------------------------------|-------------|---------------|---------------------|-----------|---------|
| nvfp4 native, old (subnormal)      | fp4  (W4A4) | subnormal     | 0.07907 +/- 0.00175 | 1.05777   | 89.041% |
| nvfp4 native, lift + .scale        | fp4  (W4A4) | normal        | 0.06880 +/- 0.00155 | 1.03824   | 89.737% |
| nvfp4 dequant control, lift+.scale | f16  (W4A16)| normal        | 0.03961 +/- 0.00134 | 1.01895   | 92.461% |
| q4_0 native                        | int8 (W4A8) | fp16          | 0.04852 +/- 0.00149 | 1.03311   | 91.737% |

Reading it:
- **Subnormal rescue lands on the native speed path:** 0.07907 -> 0.06880, **-13% KLD**, free (no new
  type; normal-grid placement + the `.scale` epilogue). The "speed XOR quality" fork above is
  half-closed - the weight rescue is no longer dequant-only.
- **As a weight format (W4A16), lifted nvfp4 beats q4_0:** 0.03961 vs 0.04852, **-18%** (top-p 92.46
  vs 91.74), and edges the old global-pow2 ceiling (0.04028) - the per-tensor LS `.scale` is a hair
  better than a uniform 2^-6 shift, and it is stored in-file so it runs on a stock binary.
- **The native gap to q4_0 is purely activations:** 0.06880 (W4A4) - 0.03961 (W4A16) = **0.02919**,
  the fp4-activation penalty (matches the +0.029 measured earlier). Native nvfp4 stays ~42% behind
  q4_0 native; the weight fix narrows but cannot close it, because the OMMA is fp4 x fp4 - there is
  no int8-activation option on the speed path.

**Net.** "Reap both native FP4 speed and precision" delivers *better* native precision
(0.079 -> 0.069, while keeping the +30% prefill), not q4_0-beating precision. To beat q4_0 you must
either drop to W4A16 (forgo the FP4 speed - then nvfp4 wins, 0.0396 vs 0.0485) or shrink the W4A4
activation error, which the weight scale cannot touch. Follow-up for a deployable type: replace the
uniform LIFT + LS `.scale` with NVIDIA-canonical per-tensor `s_global = global_amax/(448*6)`,
`s_block = (block_amax/6)/s_global` (robust per-tensor range placement, apples-to-apples vs NVIDIA's
own NVFP4).
