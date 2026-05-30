# Recovering NVFP4 W4A4 Accuracy: SmoothQuant + Hadamard

Investigation into where NVFP4's deployed (native-FP4) precision loss comes from, and how to
recover it **without** giving up the FP4 prefill throughput. Target: Qwen3-8B on Blackwell (RTX 5090,
sm_120). All KLD figures are 200-chunk wikitext-2 vs a bf16 base, `-c 512`.

## Hypothesis

NVFP4 on Blackwell runs the matmuls in two regimes:

- **W4A4** — native FP4: weights *and* activations are quantized to 4-bit E2M1 and fed to the
  block-scaled FP4 tensor cores (K=64/issue). This is the fast prefill path.
- **W4A8** — the generic int8 MMQ path (q8_1 activations, K=32), used by decode and every
  non-Blackwell GPU.

Measured, the gap is large and one-directional:

| model | W4A4 | W4A8 | W4A∞ (cuBLAS fp32) |
|---|---|---|---|
| N4_0 (bare) | 0.1179 | 0.0695 | 0.0689 |
| nvfp4-import (scale2 + Q8 emb) | 0.1135 | 0.0671 | 0.0665 |

(Q4_0 reference 0.0780; Q8_0 reference 0.0018.)

So ~0.05 Mean KLD is lost purely to **4-bit activations**, and recovering it (W4A8) costs ~42%
prefill (pp2048 ~20.5k → ~11.9k t/s; decode flat). Because 8-bit operands only exist on the K=32
MMA, W4A8 can never reclaim the FP4 K=64 throughput — it lands *below* Q4_0 on speed.

**The only way to beat Q4_0 on both speed and accuracy is to keep W4A4 (FP4 K=64) and make the
activations survive 4-bit quantization** — i.e. reshape their distribution so the per-16 E2M1 grid
fits. The standard outlier-suppression techniques are SmoothQuant (per-channel scaling) and
Hadamard rotation (incoherence processing, QuaRot/SpinQuant).

Initial guess, from the literature: the SwiGLU input to `down_proj` is the outlier hotspot, so
fixing `down_proj` should recover most of the loss.

## Methodology

To find where the W4A4 activation loss actually lives, we lift one projection at a time to
full-precision activations and measure the KLD recovered.

- **Base:** NVFP4 on the W4A4 native-FP4 path (the original Blackwell default).
- **Per-projection lift:** a runtime env `GGML_CUDA_FP4_HIPREC=<substr>` added to the CUDA matmul
  dispatch (`ggml_cuda_mul_mat`): any NVFP4 weight whose `src0->name` contains the substring is
  routed off MMQ to the cuBLAS dequant path (fp32 weights *and* fp32 activations = W4A∞). All other
  NVFP4 matmuls stay W4A4. This isolates a single projection's activation precision while leaving
  its weight quantization and every other layer untouched.
- **Metric:** Mean KLD vs bf16, `qwen3-8b-N4_0.gguf`, 200 chunks.
- **Validation:** `GGML_CUDA_FP4_HIPREC=blk` (lift everything) reproduces the global W4A∞ ceiling
  bit-for-bit (0.068874), confirming the override is sound.

Reproduce (on a W4A4-base build):

```bash
ARGS="-m qwen3-8b-N4_0.gguf -f wikitext-2-raw/wiki.test.raw \
  --kl-divergence-base kld-base/qwen3-8b-bf16-kld-base.dat --kl-divergence \
  -c 512 --chunks 200 -ngl 999 -fa on"
build-local/bin/llama-perplexity $ARGS                              # baseline W4A4
GGML_CUDA_FP4_HIPREC=ffn_down build-local/bin/llama-perplexity $ARGS # lift down_proj
GGML_CUDA_FP4_HIPREC=blk      build-local/bin/llama-perplexity $ARGS # lift all = ceiling
```

## Findings

### 1. The loss is the 4-bit quantization, not the arithmetic

| path | activations | arithmetic | Mean KLD |
|---|---|---|---|
| W4A4 | fp4 | fp4×fp4 MMA (K=64) | 0.1179 |
| W4A8 | 8-bit | int8 MMA (K=32) | 0.0695 |
| W4A∞ | fp32 | fp32 cuBLAS GEMM | 0.0689 |

The entire 0.049 is the **W4A4→W4A8 step (activation bit-width)**. Swapping int8-MMA for fp32-GEMM
(W4A8→W4A∞) — a completely different multiply order and precision — moves it <0.001. So the error is
information destroyed by 4-bit rounding (E2M1 is 8 magnitudes per sign, ~10% relative per element),
not floating-point associativity (~1e-7 relative). Reordering products (a·b·c → a·(b·c)) cannot
recover discarded bits. (The one place magnitude-ordering already matters — the per-tensor `scale2`
— is correctly factored out of the accumulation in the scale-fusion epilogue.)

### 2. Per-projection activation-loss map

Lifting each projection to W4A∞ (baseline 0.1179, ceiling 0.0689, total recoverable 0.0490):

| projection | KLD lifted | recovered | share | norm before it? |
|---|---|---|---|---|
| ffn_down | 0.1042 | 0.0137 | 28% | none |
| ffn_up | 0.1056 | 0.0123 | 25% | ffn_norm |
| ffn_gate | 0.1065 | 0.0114 | 23% | ffn_norm |
| attn q+k+v | (derived) | 0.0078 | 16% | attn_norm |
| attn_output | 0.1121 | 0.0058 | 12% | none |
| **FFN (gate+up+down)** | 0.0823 | 0.0356 | **73%** | |
| **attn (q+k+v+o)** | 0.1043 | 0.0136 | **28%** | |

(FFN + attn ≈ 101% — clean, additive decomposition.)

Two results:

- **No single hotspot.** The biggest contributor (down_proj) is only ~28%; gate, up, down are a
  ~23-28% cluster. Selective precision (lift one layer to W4A8) cannot work.
- **The down_proj hypothesis is refuted.** `gate`+`up` together (45%) lose *more* than `down_proj`
  (28%). For this model the FFN *input* projections, not the SwiGLU output, dominate.

### 3. The work cleaves ~60/40 by norm-precedence

What matters for the *fix* is whether a RMSNorm immediately precedes the projection, because that
determines whether a smoothing scale can be folded in for free:

- **~64% sits behind a norm** — `gate`/`up` (post-`ffn_norm`), `q`/`k`/`v` (post-`attn_norm`).
  Addressable by SmoothQuant with no inference-time cost.
- **~40% has no preceding norm** — `down_proj` (input is `gate(x)*up(x)`) and `attn_output` (input is
  the attention context). These need an online transform: Hadamard rotation.

## Recommendations

1. **SmoothQuant the post-norm projections first.** It is the cheapest tool (a per-channel scale
   folded into the norm weight, zero inference cost, full W4A4 speed retained) and it covers the
   single largest chunk (~64% of the recoverable loss). Highest value per unit effort.

2. **Hadamard-rotate the two no-norm projections second** (`down_proj`, `attn_output`). This is the
   only tool for activations with no norm to fold into. The FWHT primitive landed upstream
   (`c1f1e28d2`) and is the main reason this is now feasible rather than a research project.

3. **Do not ship W4A8 as the answer.** It recovers the accuracy but forfeits the FP4 prefill
   advantage — the entire reason the format exists. Keep W4A8 as an opt-in mode; pursue W4A4 +
   incoherence as the default-quality path.

**Caveat on the numbers:** the map above is the *ceiling* — what perfect activations recover per
layer. SmoothQuant only removes per-channel outliers, so it captures a fraction of its 64%; Hadamard
incoherence gets closer to the ceiling. Expect SmoothQuant alone to land N4_0 around ~0.09-0.10, and
the Hadamard pass on `down_proj`+`attn_output` to push under Q4_0's 0.078 toward the 0.069 floor.

## Implementation Strategy

### Phase 1 — SmoothQuant (producer-side, no kernel/graph change)

- Calibrate per-channel activation scales on a small text sample; pick a balance exponent (α≈0.5)
  between activation and weight magnitudes per input channel.
- Apply the migration `X·diag(s)⁻¹ · diag(s)·W`: bake `diag(s)·W` into the quantized NVFP4 weight,
  fold `diag(s)⁻¹` into the preceding RMSNorm weight so the smoothed activation is produced for free.
- Qwen3 constraint: `gate` and `up` share the `ffn_norm` output, and `q`/`k`/`v` share `attn_norm`,
  so each shared input gets **one** scale chosen jointly across the projections that consume it.
- Validate with the same KLD harness; no inference path changes, W4A4 speed unchanged.

### Phase 2 — Hadamard for the no-norm projections

The online primitive already exists: `ggml_gen_hadamard` (orthonormal, H²=I, power-of-2) +
`ggml_mul_mat_aux` (reshape → `ggml_mul_mat` against the Hadamard matrix → `GGML_HINT_SRC0_IS_HADAMARD`
→ runs as an O(n·log n) FWHT, `fwht.cuh`). The KV cache (`llama-kv-cache.cpp`) is the in-tree
reference for rotate-before-quantize.

Per projection `y = W·x`, insert an orthogonal H: `y = (W·Hᵀ)(H·x)`. `W·Hᵀ` is folded into the
NVFP4 weight offline (re-fit the per-block FP4 scales after rotation); `H·x` is the online FWHT in
front of the matmul. H is orthogonal so the result is invariant, but `H·x` spreads outliers and
quantizes to FP4 far more accurately.

- **`attn_output`** — input dim 4096 = 2¹², a clean power of 2. Easiest; build the rotate-then-FP4
  pipeline here first and validate end-to-end. **Caveat:** the FWHT CUDA kernel (`fwht.cu`) only
  covers N ∈ {64,128,256,512}; 4096 is past that, so `ggml_mul_mat_aux` falls through to a *dense*
  `mul_mat(H, x)` against the real `ggml_gen_hadamard` matrix (correct, but a 4096² GEMM per o_proj —
  not shippable speed). A shippable o_proj needs either a kernel extension to 4096 or a Kronecker
  `H₈⊗H₅₁₂`; both are follow-ups once the accuracy payoff is known.
- **`down_proj`** — input dim 12288 = 3×4096, **not** a power of 2. `ggml_gen_hadamard` asserts
  power-of-2, so this needs a grouped/Kronecker Hadamard (e.g. `H₄₀₉₆` applied across the three
  4096-blocks). This is the new generator/kernel work; tackle it after the pow2 path is proven.

### Sequencing

1. Phase 1 (SmoothQuant) — biggest single mover, cheapest, no inference cost.
2. Phase 2a — `attn_output` Hadamard (pow2), proves the pipeline.
3. Phase 2b — `down_proj` grouped Hadamard, closes the remaining loss.

Each phase is independently measurable with the per-projection KLD harness, and the success bar is
clear: land N4_0 below Q4_0 (0.078) at W4A4 speed — the point at which NVFP4 dominates Q4_0 on both
axes rather than trading against it.

## Phase 2a result — `attn_output` Hadamard, end-to-end

Built and measured on Qwen3-8B. The full `H₄₀₉₆` was used via the dense `mul_mat` fallback (above).

**Implementation** (all gated by env `GGML_NVFP4_ROTATE_OPROJ`, both sides must agree):

- *Offline fold* — `llama_fwht_rows` in `llama-quant.cpp`: an in-place normalized Walsh-Hadamard
  transform applied to each `attn_output.weight` output-channel row before NVFP4 quantization
  (`w → H·w`, then the per-block FP4 scales are re-fit on the rotated rows automatically). The
  butterfly is copied verbatim from `ggml_compute_forward_fwht` so it matches the online matrix.
  (Also wired `LLAMA_FTYPE_MOSTLY_NVFP4` into `llama_ftype_get_default_type` + the `NVFP4`
  quant-type name so `llama-quantize … NVFP4` can emit the format on this branch.)
- *Online rotation* — `oproj_hadamard` (a cached `ggml_gen_hadamard` matrix) + an `o_rot` input on
  `llm_graph_input_attn_kv`, applied via `ggml_mul_mat_aux` immediately before the o_proj matmul in
  `build_attn`. `x → H·x`, so the o_proj activation is de-outliered before fp4 quantization.

**Correctness** — three independent confirmations:

- rotated weights + online `H` → coherent text; rotated weights *without* `H` → garbage (the
  rotation is large and genuinely active, and `H·x` inverts the weight fold).
- at W4A∞ (acts fp32), rotated and unrotated are statistically identical (0.07291 vs 0.07309, Δ well
  inside noise) — the fold is bit-faithful and the orthogonal weight rotation is free at high
  precision. The entire W4A4 delta is activation de-outliering, nothing else.

**Numbers** (200-chunk wikitext KLD vs bf16; pure-NVFP4 body, Q8_0 embeddings/output):

| config | unrotated | rotated | Δ |
|---|---|---|---|
| W4A4 (native fp4) | 0.120592 ± 0.0015 | 0.118750 ± 0.0016 | −0.00184 |
| W4A∞ (acts fp32)  | 0.073093 ± 0.0012 | 0.072909 ± 0.0012 | −0.00018 (noise) |
| o_proj→W4A∞ ceiling (unrotated) | 0.116103 ± 0.0016 | | full o_proj lift = 0.00449 |

**Read:** the rotation recovers ~0.0018 KLD at W4A4 — about **41% of o_proj's full 0.0045 ceiling**,
at full W4A4 speed. But o_proj is confirmed a *small* mover: its entire ceiling is only ~9% of the
total W4A4-recoverable (~0.051), exactly as the per-projection map predicted, and the absolute gain
is ~1σ — direction and mechanism are solid, the precise fraction is not. The value delivered is the
**validated, reusable pipeline**, not the o_proj number. Point it next at the FFN (down_proj, the
28% slice with a far larger ceiling); the only non-shippable piece is the dense `H₄₀₉₆` GEMM, which a
kernel extension / Kronecker factoring removes once a high-value target justifies the work.

## Phase 2b result — `down_proj` block-diagonal Hadamard (negative)

`down_proj`'s input is 12288 = 3 × 4096, not a power of 2. The simplest grouped form — block-diagonal
`R = I₃ ⊗ H₄₀₉₆`, i.e. apply the Phase 2a `H₄₀₉₆` independently to each of the three 4096-wide
slices — reuses every Phase 2a primitive verbatim: the offline `llama_fwht_rows(buf, 4096, 3·nrows)`
naturally rotates each contiguous 4096-segment of every weight row, and the online
`ggml_mul_mat_aux(cur, H₄₀₉₆)` reshapes `[…,12288] → [4096, 3·…]` so a single `H₄₀₉₆` covers every group.
67 MB shared rotation tensor, no new generator or kernel work.

**Implementation** (env `GGML_NVFP4_ROTATE_DOWN`, both sides must agree):

- *Offline fold* — additional branch in `llama-quant.cpp` matched on `ffn_down.weight` substring,
  calling the same `llama_fwht_rows` with row length 4096 and row count `(ne0/4096)·ne1·ne2`.
- *Online rotation* — new `llm_graph_input_ffn_rot` carrying a single `H₄₀₉₆`, lazily created and
  shared across layers (looked up via `ggml_get_tensor(ctx0, "ffn_down_rot")`); `can_reuse()` returns
  true since the matrix is constant. Insert in `build_ffn` immediately before the `down` matmul.

**Correctness:** with online `H` → coherent text; without it → token salad. The fold is genuinely
active and the online rotation inverts it.

**Numbers** (Qwen3-8B, 200-chunk wikitext KLD vs bf16; pure-NVFP4 body, Q8_0 embeddings/output):

| config            | unrotated         | rotated           | Δ                  |
|-------------------|-------------------|-------------------|--------------------|
| W4A4 (native fp4) | 0.119250 ± 0.0015 | 0.152481 ± 0.0019 | **+0.0332**        |
| W4A∞ (acts fp32)  | 0.073103 ± 0.0012 | 0.103309 ± 0.0015 | **+0.0302**        |
| W4A4 − W4A∞ (act cost) | 0.0461       | 0.0492            | +0.003 (noise)     |

**Read — the rotation hurts on both axes.** Unlike o_proj, the W4A∞ invariant does *not* hold: rotated
weights at fp32 activations are +0.030 KLD *worse* than unrotated, purely a weight-quantization cost.
And the activation-quantization gap (W4A4 − W4A∞) is essentially unchanged — block-diagonal rotation
recovered **zero** of the activation loss it was meant to address. Net: about 2.5× the entire
`ffn_down` ceiling thrown away.

**Asymmetry vs o_proj.** Same FWHT primitive, same NVFP4 quantization, same group size: o_proj's W4A∞
Δ was 0.00018 (noise); `down_proj`'s is 170× larger. NVFP4's design rests on per-16 UE4M3 sub-block
scales adapting to local magnitude variation along each row. FWHT spreads each row's energy uniformly
across its 4096-element block → all per-16 scales within the block converge → the four bits of E2M1
must now span what used to be several locally-adapted ranges. `o_proj`'s trained weights are diffuse
enough that this flattening is free; `down_proj`'s are *concentrated* (specific channels carry the
SwiGLU outlier signal), and flattening them is precisely the wrong move. The same mechanism explains
the missing activation benefit: block-diagonal mixes within each 4096-block, but the channels that
need de-outliering aren't all in one block.

**Implication for cross-block mixing.** A full `H₁₂ ⊗ H₁₀₂₄` (mixing across all 12288) would worsen
the weight-flattening cost, not recover it — the conflict is between any large-radius incoherence
transform and NVFP4's per-16 scale adaptivity on a concentrated weight distribution, not between
factorizations of it. The Phase 2b path as the doc originally framed it (extend the same Hadamard
machinery to `down_proj`) is **closed within the NVFP4 weight format**. A fix for `down_proj`'s 28%
slice would need either a *smaller-radius* rotation that preserves per-16 locality (e.g. `H₁₆` inside
each NVFP4 sub-block — preserves scales but gives up almost all activation spreading), a *different
weight format* with finer-grain scales (the deferred `N4_K` two-level variant would tolerate large
rotations the way it does any flattening), or accepting the loss.

**State.** The code (`GGML_NVFP4_ROTATE_DOWN`) is left in tree as an env-gated diagnostic, default off,
mirroring the Phase 2a o_proj path; it does nothing unless the env is set on both quantize and
inference. The rotated artifact `qwen3-8b-N4_0-drot.gguf` is the measurement-only build.

## Phase 1 result — per-tensor `scale2` + per-channel SmoothQuant

The Phase 2 attempts addressed `attn_output` (12% slice, ~41% recovered) and `down_proj` (28% slice,
negative). Phase 1 — SmoothQuant on the ~64% post-norm chunk — was the largest untouched lever, and
on inspection turned out to require a producer-side step first.

### Diagnostic — UE4M3 sub-block scales sit overwhelmingly in the subnormal band

`scripts/nvfp4-subnormal-stats.py` reports, on the bare `qwen3-8b-N4_0.gguf` (no `weight_scale_2`):

```
total micro-scale codes : 434,110,464
subnormal codes (1..7)  : 400,202,766  (92.19%)
decoded scale min_nz    : 9.766e-04   (= 2^-10, the UE4M3 subnormal floor)
decoded scale max       : 2.500e-01
percentiles [.1, 1, 50, 99, 99.9] : 9.77e-4, 9.77e-4, 4.88e-3, 8.79e-3, 1.27e-2
N to rescue p0.1 : >= 3.00,   N before saturation : < 9.81,   single global N feasible: True
```

92% of N4_0's per-16 UE4M3 scales live in the coarse subnormal band (codes 1..7, fixed 2⁻¹⁰ spacing,
~10% relative error per scale) instead of the precise normal band (codes ≥ 8, 3-bit-mantissa float,
~6% per-octave). The median scale (4.9e-3) is well below normal-region start at 2⁻⁷ = 7.8e-3. The
last line is the lever's signature: a per-tensor `*2^N` shift with N ∈ [3, 9] lifts the smallest
meaningful scale above the cliff *and* keeps the largest below UE4M3 saturation. That is exactly
what `weight_scale_2` is for; the bare N4_0 quantizer doesn't emit it.

### Implementation

Both are gated, additive, no inference-side code changes (both consumers exist already):

- **`weight_scale_2` (env `GGML_NVFP4_SCALE2`)** — for every NVFP4-target weight that has a
  `build_lora_mm` consumer, emit a sibling `.scale` F32 tensor at shape `{1}` (or `{n_expert}` for
  MoE expert arrays). `s2 = amax(|W|) / (E2M1_MAX · UE4M3_MAX) = amax / 1344`. Weights are divided
  by `s2` before NVFP4 quantisation; `s2` itself is written verbatim and the existing scale-fusion
  epilogue ([[project_nvfp4_scale_fusion]]) multiplies the matmul output by it. token_embd and
  output are skipped (read via `get_rows`, no epilogue would re-apply `s2`).
- **SmoothQuant (env `GGML_NVFP4_SMOOTHQUANT`)** — for each shared-input group (FFN: `ffn_norm` →
  `{ffn_gate, ffn_up}`; ATTN: `attn_norm` → `{attn_q, attn_k, attn_v}`), compute per-channel
  `act_rms[j] = sqrt(imatrix.values[j])` (the imatrix stores mean-of-squares per input channel),
  then `s_j = (act_rms[j] / geomean(act_rms))^α`, clamped to `[0.1, 10]`. Geomean normalisation
  centres `s` on 1 so the per-tensor amax used by `scale2` stays in the same ballpark across
  α. Fold: divide the preceding `*_norm.weight` elementwise by `s`; multiply every consumer's input
  columns by `s` before NVFP4 quantisation. The migration is mathematically exact — `(X/s)·(W·s) = X·W` —
  so faithful at W4A∞ and only the activation quantisation step changes. α exposed via
  `GGML_NVFP4_SMOOTHQUANT_ALPHA` for sweeping. Requires `--imatrix`.

`scale2` runs unconditionally; SmoothQuant rides on top of it. Order in the quant loop: SmoothQuant
column scaling → rotate-oproj (if armed) → `scale2` per-tensor fold → quantize.

### Numbers (Qwen3-8B, 200-chunk wikitext KLD vs bf16; Q4_0 embeddings, Q6_K output)

scale2 alone vs the established baseline:

| config                              | W4A4 (native fp4) | W4A∞ (acts fp32) | W4A4 − W4A∞ |
|-------------------------------------|-------------------|------------------|-------------|
| N4_0 bare                           | 0.11673           | 0.06887          | 0.04786     |
| N4_0 + scale2                       | 0.11445 (−0.0023) | 0.06752 (−0.0014)| 0.04693     |

The UE4M3-subnormal share collapses 92.19% → 0.00% on the scale2 build. The W4A4 share of the lift
(−0.0023) is ~5% of the recoverable 0.0490 — small but real, matching the gap NVIDIA's calibrated
import gets from `weight_scale_2` alone. (Per the earlier `Findings §1`, the entire weight side is
~0.0689 at W4A∞ and scale precision is a sub-octave correction within that; UE4M3 normal vs
subnormal is ~2× per-scale, not 10×, so the move is bounded.)

SmoothQuant on top, α=0.5 (the SmoothQuant default; sweep below):

| config                              | W4A4 (native fp4) | W4A∞ (acts fp32) | W4A4 − W4A∞ |
|-------------------------------------|-------------------|------------------|-------------|
| N4_0 + scale2                       | 0.11445           | 0.06752          | 0.04693     |
| N4_0 + scale2 + SmoothQuant α=0.5   | 0.09463 (−0.0198) | 0.05510 (−0.0124)| 0.03953     |

Per-projection ceiling reminder: 0.0689 W4A∞, 0.0490 W4A4-recoverable, of which ~64% (≈0.031) was
the post-norm chunk SmoothQuant was supposed to address. We recovered 0.0198 of the W4A4 gap,
landing at 0.0946 — squarely in the doc's `~0.09-0.10` pre-registered range.

**The mechanism turned out different from the prediction**, and this is the most informative finding.
The doc framed SmoothQuant as attacking the activation-quantisation chunk: shrink the per-channel
activation spread → the per-16 UE4M3 *activation* sub-block scales land in the precise band. By that
framing the W4A∞ number (acts fp32, no activation quant at all) should be invariant. It is not:

  - W4A∞ moved −0.0124 (weight side improved by ~18%)
  - W4A4 − W4A∞ moved −0.0074 (activation side improved by ~16%)

So of the −0.0198 W4A4 move, ~63% is on the *weight* side and ~37% on the activation side — roughly
the *opposite* of the framing. Hypothesised mechanism: the migration multiplies outlier-activation
columns of `W` by `s_j > 1`. In transformers the outlier-activation channels tend to carry
*smaller* learned weights (the model compensates for large acts), so amplifying those columns makes
the per-channel weight distribution more uniform → tighter per-16 sub-block amax → finer fp4 grid
*on the weight side*. Per-tensor `scale2` then re-centres the whole distribution into UE4M3 normal,
compounding cleanly. The activation-side share (the originally intended payoff) is also real but
smaller. Net: SmoothQuant on the post-norm groups is a *weight-and-activation* lift, not the
activation-only lift the literature suggests.

### α sweep — broad strokes

Geomean-normalised act-driven s with α exposed via `GGML_NVFP4_SMOOTHQUANT_ALPHA`. Other knobs
(clamp range `[0.1, 10]`, RMS activation statistic, per-tensor `scale2`) held fixed.

| α              | W4A4 Mean KLD | Same-top-p | Δ vs scale2-only |
|----------------|---------------|------------|------------------|
| 0.0 (= scale2) | 0.11445       | 86.39%     |  —               |
| 0.3            | 0.09730       | 87.51%     | −0.0172          |
| **0.5**        | **0.09463**   | **87.76%** | **−0.0198**      |
| 0.7            | 0.09616       | 87.59%     | −0.0182          |
| 0.9            | 0.10111       | 87.38%     | −0.0133          |
| 1.0            | 0.10464       | 87.06%     | −0.0098          |

Parabolic, minimum exactly at the SmoothQuant default. The spread across `[0.3, 0.7]` is only
~0.003 KLD — α-tuning alone has nothing more to give from this formulation. The shallowness is
the read: the *next* axis (activation statistic, per-column weight max, or per-group α) will move
more than fine-grained α.

### State

`scale2` + SmoothQuant are env-gated, off by default; bit-for-bit unchanged behaviour with the envs
unset. Once turned on they require regenerating the GGUF (the migration bakes into the stored
weights). The 5 edits live in `src/llama-quant.cpp`; no graph or kernel changes. Reproducible
artifact: `qwen3-8b-N4_0-sq.gguf` (scale2 + SmoothQuant α=0.5).

### Where the residual loss lives

Updated picture vs Q4_0 (0.0780) at the same size and W4A4 speed:

| build                                | W4A4    | W4A∞    | gap to Q4_0 (W4A4) |
|--------------------------------------|---------|---------|--------------------|
| N4_0 bare                            | 0.11673 | 0.06887 | +0.0387            |
| N4_0 + scale2 + SmoothQuant α=0.5    | 0.09463 | 0.05510 | +0.0166            |
| Q4_0 reference                       | 0.07800 | n/a     | 0                  |

We close ~57% of the bare→Q4_0 gap at W4A4 *and* now sit *below* the original W4A∞ ceiling
(0.0689) on the weight side. The remaining +0.0166 splits roughly:

- ~0.010 in the W4A4 − W4A∞ gap = pure activation-quant residual. Untouchable by any weight-side
  knob; needs activation-side incoherence (Hadamard on `ffn_down` + `attn_output`) or a finer
  activation grid (W4A8 by `GGML_CUDA_FP4_NO_MMQ`, sacrificing FP4 K=64 speed — the trade we
  already rejected).
- ~0.006 in W4A∞ residual vs Q4_0's likely W4A∞ floor (≈0.050 for size-matched Q4_0). This is the
  4-bit *value* precision floor of E2M1 vs the 4-bit *value* precision of Q4_0's int4. Closeable
  only by a richer weight format (`N4_K` two-level — deferred, [[project_nvfp4_weights]]).

The α sweep's shallowness suggests the cheap fraction of the +0.0166 residual is already captured.
Further W4A4 movement from SmoothQuant alone likely needs the **activation statistic** swap
(max-abs vs RMS — canonical SmoothQuant) or **per-column weight max** in the denominator (proper
α=0.5, instead of geomean-normalised activation-only).

### Canonical denominator — `max|W_j|` in the denominator (negative)

Tested the per-column-weight-max axis directly (env `GGML_NVFP4_SMOOTHQUANT_WMAX`): replace the
act-only `s_j ~ act_rms_j^α` with the canonical `s_j ~ act_rms_j^α / max|W_j|^(1-α)`, where `max|W_j|`
is the joint per-input-channel weight max over the group's consumers (gate+up / q+k+v), read once in
a prelim mmap pass. RMS still stands in for `max|X_j|` (the imatrix has no max). α=0.5, same geomean
normalisation + `[0.1,10]` clamp.

| build (Qwen3-8B, 10-chunk KLD)        | W4A4     | W4A∞     | act cost (W4A4−W4A∞) |
|---------------------------------------|----------|----------|----------------------|
| scale2 + SmoothQuant α=0.5 (act-only) | 0.094625 | 0.055103 | 0.039522             |
| + canonical `max|W|` denominator      | 0.096619 | 0.055264 | 0.041355             |

**Negative: W4A4 +0.002, W∞ flat (no weight-quant gain), activation cost up.** The mechanism is the
inverse of the act-only formula's accidental win. Findings established that *outlier-activation
channels carry small weights*; the `1/max|W_j|^(1-α)` term amplifies small-weight channels, so those
outlier-act/small-weight channels get driven *harder* into the clamp ceiling and over-migrated. The
act-only geomean form is already well-matched to this weight structure — the canonical denominator
over-corrects. It also dims the `max|X|` numerator follow-up: max would enlarge the numerator for
exactly those channels, pushing them further into the clamp, so the pathology worsens rather than
resolves. Conclusion: canonical SmoothQuant is the wrong tool for NVFP4's outlier-acts-have-small-
weights structure. (The activation residual was then tried with a rotation — H₁₆ below — also
negative.) Code left in tree as the env-gated `GGML_NVFP4_SMOOTHQUANT_WMAX` diagnostic, default off
(act-only path is bit-identical when unset).

### H₁₆ micro-rotation on `down_proj` (negative) — and the corrected Phase 2b mechanism

Phase 2b argued H₄₀₉₆ killed `down_proj` by *converging the per-16 scales across a 4096-block*, and
predicted a radius-16 rotation — matching the UE4M3 sub-block exactly — would keep the flatten "inside
one scale's jurisdiction" and so be weight-cost-free. Implemented and tested (env
`GGML_NVFP4_ROTATE_DOWN16`: offline `llama_fwht_rows(buf,16,…)` per 16-input-channel block of every
`ffn_down` row; online H₁₆·x via `ggml_mul_mat_aux` + `llm_graph_input_ffn_rot`, one shared H₁₆ reused
across layers).

| build (Qwen3-8B, 10-chunk KLD)        | W4A4     | W4A∞     | act cost (W4A4−W4A∞) |
|---------------------------------------|----------|----------|----------------------|
| scale2 (unrotated)                    | 0.114446 | 0.067517 | 0.04693              |
| scale2 + H₁₆ down                     | 0.138217 | 0.094084 | 0.04413              |
| scale2 + SmoothQuant α=0.5            | 0.094625 | 0.055103 | 0.03952              |
| scale2 + SmoothQuant + H₁₆ down       | 0.121122 | 0.081645 | 0.03948              |

**Negative, and the prediction is falsified: W∞ +0.0266 — essentially identical to H₄₀₉₆'s +0.030 —
with near-zero activation gain** (act cost moves −0.0028 / 0.0000). KLD stays sane (~0.08–0.14), so the
fold + online inverse are correct; the rotation simply hurts as much at radius 16 as at 4096.

**Corrected mechanism (supersedes the Phase 2b "scale convergence" story).** The damage is not
cross-block; it is *within* each 16-block. A trained `down_proj` 16-block is concentrated (one large
weight + small ones), which is exactly what E2M1 + a per-16 UE4M3 scale represents *well*: the scale
tracks the dominant value, and the small entries carry negligible absolute error. H (any radius)
spreads the block to near-uniform magnitudes, and now all 16 must sit on E2M1's ~8-level (~1-mantissa-
bit) grid — E2M1 *hates* uniform distributions and *likes* concentrated ones. The per-16 scale
re-adapts fine; the flattened *values* are intrinsically harder for 4 bits. Radius is irrelevant
because the harm lives inside the 16-block.

**Joint conclusion with Path 1.** NVFP4's E2M1 value grid is matched to `down_proj`'s concentrated,
outlier-bearing weight structure: sharpening the migration scale (canonical denominator) over-amplifies
it, and any-radius Hadamard flattens it — both regress. The activation residual on the no-norm
projections (`down_proj`, `attn_output`) is therefore closed *within the NVFP4 weight format*; it is an
E2M1-value-grid limit, addressable only by a richer format (`N4_K` two-level, deferred). Code left in
tree as env-gated `GGML_NVFP4_ROTATE_DOWN16`, default off.

## FP8 activation probe — the 8-bit-float route is the same K=32 tier

A detour worth recording so it isn't re-attempted: keep activations at 8-bit *float* (e4m3) rather
than int8 or fp4. The precision is fully recoverable (at fine enough block granularity it reaches the
ceiling and matches int8), but it stays on the K=32 MMA — same ~Q4_0 speed tier as int8 W4A8, never
the fp4 K=64 rate that is the entire point of the format.

**Probe** (env `GGML_CUDA_FP4_FP8`, `ggml_cuda_mul_mat_nvfp4_fp8_cublas` in `ggml-cuda.cu`): forces
NVFP4 off MMQ (like `FP4_NO_MMQ`), dequantizes the weights to f16 at their exact stored 4-bit
precision, prepares the activations as f16, then runs an f16 GEMM with **f32 accumulation** (so the
probe reflects only the activation format). The env value selects the activation scheme:
`f16` = plain f16 cast (control, isolates the e4m3 cost from the harness = ceiling); `16`/`32` = e4m3
round-trip with a per-blk amax scale along K (`quant_dequant_e4m3_blocks_kernel`; 16 = NVFP4's native
sub-block granularity and what an `mxf8f6f4` block_scale carries, 32 = q8_1's); any other value (e.g.
`1`) = e4m3 per-row (per-token, `quant_dequant_e4m3_rows_kernel`). Precision-faithful (a native
`mxf8f6f4` kernel accumulates identically) but NOT a speed path (rides cuBLAS; the point is the KLD).
`cublasGemmEx` with `CUDA_R_8F_E4M3` was tried first and rejected every config ("unsupported
parameter"), which is why the probe round-trips through f16 instead of a real FP8 GEMM.

**Numbers** (qwen3-8b-N4_0, wikitext KLD vs bf16; same fixed chunk set as the tables above):

| activation                  | Mean KLD | scale granularity | MMA  |
|-----------------------------|----------|-------------------|------|
| W4A∞ (fp32 acts, ceiling)   | 0.068872 | —                 | cuBLAS |
| **16-bit f16 acts (`=f16`)** | 0.068861 | —                | (K=16) |
| int8 W4A8 (Findings §1)     | 0.0695   | per-32 along K    | K=32 |
| **fp8 e4m3, per-16 (`=16`)** | 0.070077 | per-16 along K   | (K=32) |
| **fp8 e4m3, per-32 (`=32`)** | 0.071238 | per-32 along K   | (K=32) |
| **fp8 e4m3, per-row (`=1`)** | 0.072982 | per-token        | (K=32) |
| W4A4 native                 | 0.1167   | per-16 (fp4)      | K=64 |
| Q4_0 reference              | 0.0780   | —                 | —    |

The `=f16` control lands on the ceiling to 1e-5 (0.068861 vs 0.068872) — the harness is sound, and
**any activation precision ≥16-bit is free** (the weight-only ceiling). So the fp8 numbers are
genuinely the e4m3 cost.

**Read — fp8 precision is a clean function of block granularity, and the gap is the block size, not
the format.**

1. The e4m3 residual vs the ceiling halves with each granularity step: per-row +0.0041 → per-32
   +0.0024 → per-16 **+0.0012**. The residual lives *across channels within a token* (per-row barely
   beats per-tensor), so it is K-block size, not per-token spread, that matters — exactly what
   shrinking 32→16 along K attacks.
2. At per-16 — NVFP4's native granularity — fp8 (0.0701) **matches int8 per-32 (0.0695)** to within
   noise and sits ~0.0012 above the ceiling. e4m3 is a hair coarser per-element than int8, but a 2×
   finer block more than compensates. So fp8 with native block scaling is ceiling-grade and clears
   Q4_0 (0.0780) comfortably. (This corrects the earlier "fp8 is worse than int8" — it was a
   granularity artifact of the per-row probe.)
3. Speed is the wall: 8-bit operands only exist on the `m16n8k32` MMA, 16-bit on `m16n8k16` — neither
   reaches the fp4 K=64 rate (measured fp4-native pp512 ~14.4k vs the cuBLAS-dequant path ~3.4k t/s).
   A native `mxf8f6f4` kernel with per-16 block_scale would be **ceiling-grade precision at the K=32
   tier** — a good operating point, but the *same* tier as int8 W4A8 (which already exists), so no
   clear win over int8 to justify the kernel.

So the FP8 route lands exactly where W4A8 did: 8-bit activations solve precision (fp8 per-16 = int8 =
near-ceiling, both beat Q4_0) and forfeit the FP4 throughput. **A native `mxf8f6f4` kernel is
defensible on precision but offers no speed/precision edge over int8 W4A8 — not worth building.** The
only route to Q4_0-precision-at-W4A4-speed remains SmoothQuant + Hadamard (Phase 1 / Phase 2). The
probe is left in tree as an env-gated diagnostic, default off, mirroring the other `FP4_*` envs.

## Tooling

- Diagnostic: `GGML_CUDA_FP4_HIPREC=<substr>` (per-name W4A∞ lift), `GGML_CUDA_FP4_NO_MMQ=1` (global
  W4A∞), `GGML_CUDA_FP4_FP8=<mode>` (fp8-activation probe through an f32-accumulate f16 GEMM,
  accuracy-only: `16`/`32` = e4m3 per-blk-along-K, `1` = e4m3 per-row, `f16` = f16 control = ceiling),
  all in the CUDA matmul dispatch. Require a W4A4-base build.
- Quantize-side folds (all off by default, all in `src/llama-quant.cpp`):
  - `GGML_NVFP4_SCALE2=1` — emit per-tensor `.scale` (`weight_scale_2`) for every NVFP4 weight
    with a `build_lora_mm` consumer. No imatrix required.
  - `GGML_NVFP4_SMOOTHQUANT=1` — per-channel migration for post-norm groups. Requires `--imatrix`.
  - `GGML_NVFP4_SMOOTHQUANT_ALPHA=<float>` — overrides α (default 0.5). For sweep experiments.
  - `GGML_NVFP4_ROTATE_OPROJ=1` / `GGML_NVFP4_ROTATE_DOWN=1` — Phase 2a/2b Hadamard folds; need
    matching env on inference.
- KLD board: `scripts/nvfp4-8b-kld.sh`; coherence gate: `nvfp4-smoke.sh`.
- Distributional diagnostic: `scripts/nvfp4-subnormal-stats.py <gguf>` reports the UE4M3 sub-block
  scale histogram + global-N feasibility (the readout that flagged the `scale2` lever).
