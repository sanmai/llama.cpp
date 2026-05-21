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

**Takeaway.** The MSE search is a worthwhile, free improvement to the *existing* NVFP4
reference quantizer (~9% lower KLD for any nvfp4 weight conversion or KV write, and it removed
CPU/CUDA encoding drift). It does not make nvfp4 the preferred 4-bit format. Caveats: single
small model; no imatrix (nvfp4 ignores it - q4_0/q4_K with imatrix would widen their lead).

## Search depth: +/-2 vs +/-3 (Qwen3-1.7B, vs same f16 base)

| search depth        | Mean KLD            | quant time |
|---------------------|---------------------|------------|
| +/-2 (committed)    | 0.173927 +/- 0.00069 | 9.8s       |
| +/-3                | 0.177830 +/- 0.00069 | 14.1s      |

+/-3 is **worse** by +0.0039 KLD (~5.7 sigma), at ~1.4x the cost. This is not noise and is the
informative result: a min-error search can only *lower* per-block SSE with more candidates, so
+/-3 yields lower unweighted SSE but HIGHER KLD. The search optimizes the wrong objective -
unweighted SSE over-weights the many small values, so a wider scan drops the scale to fit the
bulk better while distorting the few large values that drive the output. Conclusion: do not
widen; the depth is already past where the unweighted proxy diverges from output quality. This
directly motivates importance-weighting the error (x^2 / imatrix) instead of more depth.

## x^2-weighted error (+/-2)

_(pending)_
