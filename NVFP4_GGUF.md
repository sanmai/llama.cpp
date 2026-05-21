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

_(to be filled in)_
