# Spec: NVFP4 KV cache on the CUDA backend

## Goal

Allow `-ctk nvfp4 -ctv nvfp4` (`GGML_TYPE_NVFP4`) as a flash-attention KV cache
type on the CUDA backend. NVFP4 already exists in ggml as a weight-quantization
type; this work adds the write path, the decode read path, and the dispatch glue
needed to use it for the K/V cache.

## Hypothesis to test

NVFP4 stores 4.5 bits/element with a per-16 FP8 (UE4M3) micro-scale, versus
q4_0's single int scale per 32. The bet is a q4_0-class footprint with materially
better numeric fidelity for the KV cache. Whether that reaches q8_0-class quality
*for KV* (which is more outlier-sensitive than weights) is empirical and must be
measured, not assumed.

## Non-goals

- No native Blackwell FP4 tensor-core use. The KV path does not feed FP4 MMA (see
  Constraints). This is a storage/bandwidth change, not a compute one.
- No new ggml type. NVFP4 (`= 40`) already exists; this is a new *use* of it, which
  is a far lower maintenance bar than `CONTRIBUTING.md`'s "adding a quant type".
- No CPU-backend NVFP4 KV work. CUDA only.

## Background: why NVFP4 is KV-viable

Block layout, `ggml/src/ggml-common.h:211-216`:

```c
#define QK_NVFP4     64
#define QK_NVFP4_SUB 16   // sub-block size for per-group scales
typedef struct {
    uint8_t d[QK_NVFP4/QK_NVFP4_SUB]; // 4x UE4M3 (FP8) scales, one per 16-elem sub-block
    uint8_t qs[QK_NVFP4/2];           // 32x packed 4-bit E2M1 codes
} block_nvfp4;                        // 36 bytes / 64 elems = 4.5 bits/elem
```

The load-bearing property: ggml's NVFP4 is **block-local with no per-tensor scale**.
`quantize_row_nvfp4_ref` (`ggml/src/ggml-quants.c:342-375`) computes an independent
amax -> UE4M3 per 16-element sub-block; the "real" NVFP4 second-level per-tensor
FP32 amax was dropped. Incremental cache writes quantize one row chunk in isolation,
exactly like q8_0/q4_0. A two-level scheme would have made KV writes impossible.

Head-dim alignment is free: `QK_NVFP4 == 64` and the VEC kernel already requires
`D % 64 == 0`, so head dims 64/128/256 are an integer number of nvfp4 blocks per
row. None of the partial-block trouble q8_0 (block 32) hits at head dim 80/112.

## What already exists

- `block_nvfp4` type + traits, reference quantize/dequantize (`ggml-quants.c`).
- CUDA dequant: `dequantize_block_nvfp4` / `dequantize_row_nvfp4_cuda`
  (`ggml/src/ggml-cuda/convert.cu:621-657`), registered in `ggml_get_to_fp16_cuda`
  and `..._to_fp32_cuda` (`convert.cu:759, 814`).
- NVFP4 weight matmul via MMQ, with a native Blackwell path and a generic DP4A
  fallback (`ggml/src/ggml-cuda/mmq.cuh:148-152, 247-251, 945-955`). Not used here,
  but confirms the LUT/scale arithmetic is already worked out and can be borrowed.

## Two fattn regimes (dispatch: `ggml/src/ggml-cuda/fattn.cu:457-478`)

The CUDA dispatch routes quantized KV to two kernels by batch width `Q->ne[1]`:

- **Prefill / large batch (`Q->ne[1] > 2`) -> MMA_F16.** The MMA kernel consumes
  only `half2` tiles. `launch_fattn` materializes K/V to an f16 scratch buffer via
  `ggml_get_to_fp16_cuda(type)` before the kernel (`fattn-common.cuh:965-1023`;
  MMA passes `need_f16_K=need_f16_V=true`, `fattn-mma-f16.cuh:1954`). Since the
  NVFP4 dequant is already registered, **this path works today** once the type
  clears the dispatch gate.
- **Decode / small batch (`Q->ne[1] <= 2`) -> VEC.** The only path that reads the
  quantized cache in place, via `vec_dot_fattn_vec_KQ_*` (K) and `dequantize_V_*`
  (V), dispatched in `get_vec_dot_KQ` / `get_dequantize_V`
  (`fattn-common.cuh:581, 603`). No NVFP4 variants exist. This is the bulk of the
  work and is mandatory for token generation.

## Design / staging

Prove the quality hypothesis on the free MMA path before writing any VEC kernels.

### Phase 1 - measurable end to end via MMA only

1. **Write path (cpy).** Add `cpy_blck_f32_nvfp4` in
   `ggml/src/ggml-cuda/cpy-utils.cuh` (port the per-sub-block amax/UE4M3 loop from
   `quantize_row_nvfp4_ref`), `ggml_cpy_f32_nvfp4_cuda` in
   `ggml/src/ggml-cuda/cpy.cu`, and wire into `ggml_cuda_cpy`. Caveat: the generic
   `cpy_f32_q<blck, QK>` template assumes one scale per block; nvfp4 has 4 sub-scales
   per 64-block, so the per-block writer needs the sub-block loop (cannot reuse the
   single-scale template as-is).
2. **CLI allowlist.** Add `GGML_TYPE_NVFP4` to `kv_cache_types`
   (`common/arg.cpp:399`). One line.
3. **Dispatch gate.** Add `GGML_TYPE_NVFP4` to the K-type switch
   (`fattn.cu:430-446`) and force it onto MMA: do not return `BEST_FATTN_KERNEL_VEC`
   for nvfp4 until phase 2 (otherwise decode hits a missing VEC instance and aborts).
4. **Measure.** Perplexity + KL divergence vs q8_0 / q4_0 / f16 on a real model.
   This is the go/no-go gate; if nvfp4 KV does not beat q8_0 on quality-per-byte,
   stop here.

### Phase 2 - decode bandwidth via VEC kernels (only if phase 1 passes)

5. **`dequantize_V_nvfp4<T, ne>`** in `fattn-common.cuh` - trivial; mirror
   `dequantize_V_q4_0` plus the existing `dequantize_block_nvfp4`: code -> E2M1 LUT
   x UE4M3 sub-scale.
6. **`vec_dot_fattn_vec_KQ_nvfp4`** in `fattn-common.cuh` - the hard one. Q is
   quantized to Q8_1 in-kernel. Start with a float MAC (model on the bf16 path) for
   correctness; optimize to a DP4A + LUT path (model on mxfp4's
   `get_int_from_table_16` + per-sub-block scale) once correct.
7. **Register** both in `get_vec_dot_KQ` / `get_dequantize_V`.
8. **Instances + dispatch.** Add `FATTN_VEC_CASES_ALL_D(NVFP4, NVFP4)`
   (`fattn.cu:265-326`), the
   `template-instances/fattn-vec-instance-nvfp4-nvfp4.cu` file (regenerate via
   `template-instances/generate_cu_files.py`), and let the dispatch return
   `BEST_FATTN_KERNEL_VEC` for nvfp4 at `Q->ne[1] <= 2`. Almost certainly gate the
   instances behind `GGML_CUDA_FA_ALL_QUANTS` to keep default compile time sane (the
   default build only compiles `{f16, q4_0, q8_0, bf16}` KV combos).

## Constraints and gotchas

- **FP4 tensor cores are irrelevant to KV.** VEC decode is per-thread software
  unpack; MMA prefill dequantizes nvfp4 -> f16 and runs the f16 MMA. The 5090's
  5th-gen FP4 path (and the `120a` build target) buys nothing here.
- **Prefill cost.** As with any quantized KV, MMA prefill transiently materializes
  f16 K+V scratch (~3.5x the nvfp4 footprint) and runs a dequant pass per micro-batch.
  All savings are at decode; prefill is pure overhead.
- **K must equal V type** unless `GGML_CUDA_FA_ALL_QUANTS` (`fattn.cu:424-428`).
  nvfp4/nvfp4 is the target combo; mixed combos are out of scope.
- **Block-local scale only.** Confirmed above; relied on for incremental writes.

## Validation

- `./build/bin/test-quantize-fns -v` - already exercises NVFP4 (auto-enumerates
  registered types): quantize->dequantize roundtrip, ref-vs-impl, dot-product error.
  Validates the *reference quantizer* reused by the cpy write path, not the CUDA KV
  kernels.
- `test-backend-ops` - add a FLASH_ATTN_EXT case with K/V = nvfp4 to check the CUDA
  kernels against the CPU reference. Required for phase 2; useful smoke for phase 1.
- `llama-perplexity` + KL divergence vs q8_0 / q4_0 / f16 - the actual hypothesis
  test (phase 1 gate).
- `llama-bench` decode t/s vs q8_0 - the phase 2 payoff (bandwidth at tg).

## Open questions / risks

- Does the float-MAC `vec_dot` give acceptable decode throughput, or is the DP4A+LUT
  path needed before nvfp4 decode is competitive with q8_0?
- Compile-time / binary-size cost of the nvfp4 VEC instances under
  `GGML_CUDA_FA_ALL_QUANTS`; confirm the gating keeps the default build unchanged.
- KV sensitivity: per-16 micro-scale helps with outliers, but K (pre-softmax) and V
  may want different precision. Worth measuring K=nvfp4/V=q8_0 etc. if `FA_ALL_QUANTS`
  is on, even though nvfp4/nvfp4 is the headline.

## Touch list

| Area            | File(s)                                                        |
|-----------------|----------------------------------------------------------------|
| Write (cpy)     | `ggml/src/ggml-cuda/cpy-utils.cuh`, `ggml/src/ggml-cuda/cpy.cu` |
| Decode read     | `ggml/src/ggml-cuda/fattn-common.cuh`                          |
| Dispatch        | `ggml/src/ggml-cuda/fattn.cu`                                  |
| Instances       | `ggml/src/ggml-cuda/template-instances/` (+ `generate_cu_files.py`) |
| CLI allowlist   | `common/arg.cpp`                                               |
| Tests           | `tests/test-backend-ops.cpp`                                   |

Phase-1 reality differed from the plan above: the KV write path is `SET_ROWS`, not `CPY`
(`set-rows.cu` + its `supports_op`), and the MMA read needs the *non-contiguous* dequant
`dequantize_block_nvfp4_nc` (`convert.cu`), since cache views are not contiguously allocated.

## Results (phase 1, 2026-05-21)

Model OLMoE-1B-7B-0924-Instruct Q4_K_M, full `wiki.test.raw`, ctx 512, b4096/ub2048.
Base = f16 KV on master (`kld-base.dat`); KV type is the only variable. KLD = divergence
from the f16-KV reference distribution.

| KV type | bits/elem | Mean KLD            | PPL ratio | Median KLD | RMS Dp | Same top-p |
|---------|-----------|---------------------|-----------|------------|--------|------------|
| q8_0    | 8.5       | 0.00658 +/- 0.00005 | 1.00184   | 0.00251    | 2.15%  | 96.32%     |
| q4_0    | 4.5       | 0.02685 +/- 0.00016 | 1.00865   | 0.01238    | 4.36%  | 92.46%     |
| nvfp4   | 4.5       | 0.06468 +/- 0.00038 | 1.02891   | 0.03105    | 6.75%  | 88.35%     |

The table above is the **single-code** quantizer (one UE4M3 per sub-block, no search).
At that point nvfp4 was a no-go: same footprint as q4_0 (4.5 bpw) but ~2.5x worse, ~10x
worse than q8_0.

### K-vs-V asymmetry (single-code)

| K / V       | Mean KLD | Same top-p |
|-------------|----------|------------|
| nvfp4/f16 (K quantized) | 0.02953 | 92.12% |
| f16/nvfp4 (V quantized) | 0.04847 | 89.85% |
| q4_0/f16  (K quantized) | 0.02252 | 93.00% |
| f16/q4_0  (V quantized) | 0.01577 | 94.26% |

nvfp4 is a fine *K* quantizer (~q4_0 tier) but a poor *V* quantizer (3x worse than q4_0 on V).
The deficit is concentrated in V - where quantization error feeds the output directly, vs K
error that softmax damps.

### 5-candidate scale search

Refining each sub-block's UE4M3 code over `first +/- {0,1,2}` by min squared reconstruction
error (port of `quantize_mmq_nvfp4`, `quantize.cu:116-148`):

| K / V       | KLD single-code | KLD search | change |
|-------------|-----------------|------------|--------|
| nvfp4/nvfp4 | 0.06580         | 0.03088    | -53%   |
| nvfp4/f16   | 0.02953         | 0.02339    | -21%   |
| f16/nvfp4   | 0.04847         | 0.01761    | -64%   |

As predicted, the search helps V (`f16/nvfp4`, -64%) far more than K (`nvfp4/f16`, -21%).

**Revised verdict.** With the search nvfp4/nvfp4 (0.0309, top-p 91.8%) draws level with
q4_0/q4_0 (0.0282, 92.2%) at equal bits - viable, but it does not *beat* q4_0 and is still
~4x q8_0. So the search is mandatory (without it, dead), but quality alone does not justify a
new KV path over the existing q4_0. Mechanism unchanged: still single-level (the search tightens
each block's scale; it cannot recover the dropped per-tensor scale).

**CPU reference mirrored.** The search initially made CUDA quantize better than - and thus
differ from - the un-searched CPU reference, failing `test-backend-ops -o SET_ROWS` nvfp4
10/147. Resolved by porting the same search into `quantize_row_nvfp4_ref` (`ggml-quants.c`):
SET_ROWS back to 147/147 (2/2 backends), and the reference quant itself improved -
`test-quantize-fns` nvfp4 abs error 0.00234 -> 0.00203, dot-product error 0.01977 -> 0.00242
(8x). This also upgrades CPU inference and nvfp4 *weight* conversion.

### Second model: Qwen2.5-7B (dense), with search

Full `wiki.test.raw`, own f16-KV base. KV type is the only variable.

| K / V       | Mean KLD | Same top-p |
|-------------|----------|------------|
| q8_0/q8_0   | 0.00178  | 98.02%     |
| q4_0/q4_0   | 5.50890  | 11.65%     |
| nvfp4/nvfp4 | 7.00135  | 5.27%      |
| nvfp4/f16 (K) | 6.99750 | 5.31%     |
| f16/nvfp4 (V) | 0.00442 | 96.75%    |
| q4_0/f16  (K) | 5.48742 | 11.75%    |
| f16/q4_0  (V) | 0.00405 | 96.89%    |

The K-vs-V asymmetry **inverts and goes catastrophic** vs OLMoE: quantizing K to 4 bits
destroys the model (nvfp4/f16 7.0, q4_0/f16 5.5; ~5-12% top-p), while quantizing V is nearly
free (nvfp4 0.0044, q4_0 0.0040; ~97%). Symmetric KLD is driven entirely by K (nvfp4/nvfp4 ~
nvfp4/f16, q4_0/q4_0 ~ q4_0/f16). This is the Qwen2 K-outlier problem: outlier K channels
dominate the 4-bit block scale; only q8_0 K (0.0018) has the headroom. nvfp4's finer grouping
does not save K - it is *worse* than q4_0 there (7.0 vs 5.5).

### Conclusion (two models)

nvfp4 KV **tracks q4_0 and never beats it** - as V-quant (OLMoE 0.0176 vs q4_0 0.0158; Qwen
0.0044 vs 0.0040) and as K-quant (modest on OLMoE, catastrophic on both 4-bit formats on Qwen).
So a symmetric nvfp4/nvfp4 KV type is not worth adding: same footprint as q4_0, never better,
worse failure mode on hard models. The real, format-agnostic lesson: KV quant must be
**asymmetric and K-precision-biased** - V is cheap at 4 bits everywhere, K wants >= 8 bit
(mandatory on Qwen). The useful config is `K=q8_0 / V=q4_0|nvfp4` (needs `GGML_CUDA_FA_ALL_QUANTS`).

### K=q8_0 V-quant shootout (K held safe)

| Model      | V=nvfp4 KLD | V=q4_0 KLD | V=nvfp4 top-p | V=q4_0 top-p | V=nvfp4 PPLr | V=q4_0 PPLr |
|------------|-------------|------------|---------------|--------------|--------------|-------------|
| OLMoE      | 0.01766     | 0.01597    | 93.81%        | 94.15%       | 1.00347      | 1.00386     |
| Qwen2.5-7B | 0.00519     | 0.00477    | 96.51%        | 96.70%       | 1.00382      | 1.00583     |

With K fixed at q8_0, **q4_0 beats nvfp4 as the V quant on both models** (KLD ~9-11% lower,
top-p higher). nvfp4's PPL ratio is marginally lower on both (matches f16 corpus PPL slightly
better while its per-token distribution sits *further* from f16) - 3rd-decimal, not the KV
faithfulness metric. Final: nvfp4 KV has no edge over q4_0 in any config tested. The keeper is
`K=q8_0 / V=q4_0`: on Qwen it takes catastrophic 4-bit-K (5.5-7.0) down to 0.0048, ~3x of full
q8_0/q8_0 (0.0018) at half the V storage.

### Practical guidance: do not 4-bit the K cache on Qwen2-family

The single most actionable result, independent of nvfp4. On Qwen2.5-7B, a 4-bit **K** cache
is destructive regardless of format - it is the K side alone, not "4-bit KV":

| flags | KLD | Same top-p | note |
|-------|-----|------------|------|
| `-ctk q4_0  -ctv q4_0`  | 5.51   | 11.6% | murder - agrees with f16 ~1 token in 9 |
| `-ctk q4_0  -ctv f16`   | 5.49   | 11.8% | same damage; it is all K |
| `-ctk nvfp4 -ctv nvfp4` | 7.00   | 5.3%  | nvfp4-K even worse than q4_0-K |
| `-ctk f16   -ctv q4_0`  | 0.0040 | 96.9% | V at 4-bit is nearly free |
| `-ctk q8_0  -ctv q4_0`  | 0.0048 | 96.7% | the safe asymmetric config |
| `-ctk q8_0  -ctv q8_0`  | 0.0018 | 98.0% | reference low-loss |

Cause: Qwen2's K-outlier channels hijack the 4-bit block scale; only q8_0 K has the headroom.
Rule of thumb: keep **K at q8_0 (or f16)**; V quantizes to 4 bits cheaply. Caveats: this is the
KV *cache*, not weights (q4 weights are fine - the q4_0-weight model with f16 KV is the 0.000
baseline); measured on one model but consistent with the documented Qwen2 family behavior; OLMoE
(MoE) did *not* show it, so it is architecture-dependent, not universal.
