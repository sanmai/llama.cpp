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

## Tooling

- Diagnostic: `GGML_CUDA_FP4_HIPREC=<substr>` (per-name W4A∞ lift), `GGML_CUDA_FP4_NO_MMQ=1` (global
  W4A∞), both in the CUDA matmul dispatch. Require a W4A4-base build.
- KLD board: `scripts/nvfp4-8b-kld.sh`; coherence gate: `nvfp4-smoke.sh`.
