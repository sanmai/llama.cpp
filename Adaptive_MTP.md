# Adaptive MTP Draft Depth

Working notes on making the MTP draft depth (`--spec-draft-n-max`) adapt to the observed
acceptance rate, instead of a single fixed value. Draft models: gemma-4-31B-it (MTP head,
shared-memory path) and Qwen3.6-27B-MTP (single-head, separate-context path).

## Hypothesis

Speculative decoding emits, per target step, roughly `(1 - a^(D+1)) / (1 - a)` tokens for
draft length `D` and per-token acceptance `a`. The marginal token from extending `D -> D+1`
is `a^(D+1)`; extending pays as long as that exceeds `draft_step_cost / target_decode_cost`.
Since `a^(D+1)` shrinks with `D`, the optimal draft depth **grows as `a -> 1`**.

Agentic/coding output is **bimodal in `a`**: long verbatim-echo stretches (citing context,
emitting diff context lines, reproducing a function) at `a ~ 0.95-1.0`, punctuated by
reasoning / novel tokens (the actual edit decision, control-flow, prose) at `a ~ 0.5-0.7`.
A single fixed `n_max` cannot be optimal for both. Therefore adapting `n_max` to recent
acceptance should beat any fixed value on realistic workloads.

## Background: what bounds MTP drafting today

Code refs are against `common/speculative.cpp` (struct
`common_speculative_impl_draft_mtp`) unless noted.

- The draft loop (`draft()`) already terminates on two conditions per step:
  - hard cap `params.n_max <= result.size()` (the configured `--spec-draft-n-max`);
  - per-token confidence `cur_p->data[0].p < params.p_min` (early stop).
- `p_min` defaults to **0.0** (`common/common.h:329`), i.e. the confidence gate is *off* by
  default: drafts always run to the full `n_max`. `--spec-draft-p-min` turns it on.
- gemma-4 takes the **shared-memory** path (`is_mem_shared = true`), so `chain_heads = false`
  and `n_max` is *not* clamped to the trained head count — it can be raised freely. Draft
  tokens are added at the **same position** (single head reused autoregressively), so the
  head is not trained for deep unroll; tail acceptance may decay faster than a real draft
  model.
- Qwen3.6-27B-MTP takes the **single-head, non-shared** path (`is_mem_shared = false`,
  `n_mtp_layers = 1`): draft tokens get proper growing-KV positions (`dp.n_past + i + 1`),
  which may unroll deeper more gracefully than gemma's same-position reuse.
- The speculative context (and the MTP impl's `n_max`/`p_min`, copied at construction) is
  built **once** at server init (`tools/server/server-context.cpp:1359`). Per-request
  `speculative.n_max`/`p_min` never reach the frozen impl. **Consequence: every config
  requires a server restart** — a per-request sweep would silently return identical numbers.

## Methodology

- Draft/target: `unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL` and
  `unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K`, MTP head auto-discovered. `--spec-type draft-mtp`.
- Harness: `tmp/nmax-pmin-sweep.sh`. Restarts the server per `(p_min, n_max)` config by
  reusing the known-good launcher scripts and appending CLI overrides via their trailing
  `"$@"` — CLI wins over the env vars the launchers hardcode
  (`common/arg.cpp:650`, "will be overwritten by command line argument").
- Prompt: a ~32k-token context (the file `common/speculative.cpp`) plus an instruction that
  targets the same region (`struct common_speculative_impl_draft_mtp`) under three edit
  tasks, so **recitation content is held constant and only the edit type varies**:
  - `refactor`  - extract a helper method: structural, clustered edits.
  - `replace`   - substitute every `float` -> `float32`: mechanical, sparse edits; the
    substituted identifier is the density knob that dials `a` as an independent variable.
  - `autotype`  - replace every `auto` with the deduced concrete type (best guess when
    uncertain): edits that require reasoning.
- Generation: `n_predict = 4096` (long responses sustain the echo regime and average out
  noise), `temperature 0.6`, `seed 1234`, `cache_prompt true`.
- Metrics scraped from the server log (guaranteed INFO lines
  `server-context.cpp:576` and `:629`): `tg` (throughput, tok/s), `acc` (draft accept
  fraction), `mean_len` (accepted tokens per target step). Single-shot per config.

Caveats: single-shot, so treat sub-10% gaps as noise. Edit correctness is *not* graded — a
wrong guess still emits a divergence at the right spot, so the acceptance-regime measurement
holds regardless. `replace` scatters divergences (geometric run-lengths); `refactor`
clusters them (bursty) — different run-length statistics, both useful.

## Results

### gemma-4-31B, three edit tasks (n_predict 4096)

Best config and baseline per task (full tables below):

| task     | edit type       | n=4 baseline | a at n=4 | best config | best tg | mean_len | gain |
|----------|-----------------|--------------|----------|-------------|---------|----------|------|
| replace  | mechanical      | 120 t/s      | 0.97     | p0.85 / n12 | 226     | 11.2     | +89% |
| autotype | reasoning       | 111 t/s      | 0.90     | p0.80 / n12 | 174     | 8.6      | +56% |
| refactor | structural      | 112 t/s      | 0.86     | p0.75 / n8  | 159     | 6.9      | +42% |

Full tables (`tg` tok/s):

```
refactor                         replace                          autotype
p_min n_max  tg     acc   mlen   p_min n_max  tg     acc   mlen   p_min n_max  tg     acc   mlen
0.75  4      105.9  0.838 4.16   0.75  4      119.6  0.966 4.82   0.75  4      111.4  0.909 4.47
0.75  5      110.6  0.849 4.96   0.75  5      119.5  0.925 5.47   0.75  5      112.3  0.878 5.14
0.75  6      107.4  0.791 5.24   0.75  6      130.4  0.968 6.71   0.75  6      117.9  0.882 5.97
0.75  8      159.0  0.807 6.86   0.75  8      191.8  0.940 8.21   0.75  8      154.6  0.818 6.74
0.80  4      112.2  0.924 4.56   0.80  4      118.2  0.969 4.78   0.80  4      105.9  0.879 4.29
0.80  6       99.98 0.762 4.87   0.80  6      126.1  0.943 6.47   0.80  6      112.0  0.862 5.73
0.80  8      148.3  0.797 6.40   0.80  8      189.4  0.932 8.12   0.80  8      140.1  0.781 6.21
0.80  12     156.0  0.707 7.59   0.80  12     204.0  0.860 10.07  0.80  12     174.2  0.781 8.61
0.85  6      108.5  0.826 5.29   0.85  6      127.1  0.956 6.47   0.85  6      108.3  0.844 5.45
0.85  8      146.0  0.821 6.42   0.85  8      187.0  0.951 8.05   0.85  8      134.6  0.781 5.81
0.85  12     153.6  0.760 7.54   0.85  12     226.3  0.912 11.17  0.85  12     164.5  0.785 7.85
0.90  8      142.3  0.829 6.31   0.90  8      183.6  0.937 7.84   0.90  8      135.5  0.810 5.82
0.90  12     146.4  0.741 7.11   0.90  12     209.6  0.895 10.02  0.90  12     146.9  0.748 6.96
```

Reads:
- The three tasks form a clean **`a`-ladder** (replace 0.97 > autotype 0.90 > refactor 0.86),
  and the optimal depth shifts **right** as `a` rises — the hypothesis, demonstrated.
- `replace` at n=12 still has `a = 0.91` and `mean_len = 11.2`; the curve has not turned
  over, so this workload wants `n_max > 12` (see ceiling probe).
- Long responses corrected an earlier read: a 600-token run said "saturates at n=8"; with
  4096-token responses n=12 clearly beats n=8 everywhere (most on replace, 226 vs 189),
  because short generations do not stay in the high-`a` regime long enough to amortize deep
  drafts.
- `p_min` is forgiving in 0.75-0.85; 0.90 starts clipping useful drafts.

### Mixed / annotated recitation (gemma)

`mixed` = reproduce the struct verbatim but insert a novel one-line comment before each
method: reason and echo finely interleaved, the cadence of an agent annotating code.

```
p_min n_max  tg     acc    mean_len
0.80  4      117.5  0.957  4.74
0.80  8      177.8  0.907  7.71
0.80  12     194.9  0.845  9.60
0.85  8      174.5  0.915  7.54
0.85  12     194.0  0.861  9.60
0.90  12     190.0  0.877  9.46
```

The four tasks now form the ladder (acc, at p0.80/n12): replace 0.86 > mixed 0.845 >
autotype 0.78 > refactor 0.71. **Annotated recitation stays essentially in the echo
regime** (acc 0.90-0.96, close to `replace`) — the inserted comments are a small token
fraction, so the stream never leaves the fast regime and still wants deep drafts (n=12 best,
195 t/s). This is the diff-style case (mostly-context edits) and it strengthens `n=12` as a
static default: realistic edit-shaped output is echo-dominated. Caveat: aggregate does not
show temporal switching, and a *write-new-logic* task (little to echo) would land lower than
this; annotated recitation is the favorable end of "mixed".

### Echo-regime ceiling: replace at n_max 12/16/24 (gemma)

```
p_min n_max  tg     acc    mean_len
0.80  12     205.9  0.860  10.07
0.80  16     215.7  0.833  12.03
0.80  24     202.3  0.702  12.97
0.85  12     226.4  0.912  11.17
0.85  16     234.4  0.867  13.24   <- gemma echo peak
0.85  24     196.4  0.723  12.41
```

gemma's echo ceiling is **n ~= 16** (peak 234 t/s at p0.85). n=24 **regresses**: acceptance
collapses (0.87 @ 16 -> 0.72 @ 24) even though mean_len keeps rising — the deep drafts are
generated and then rejected, wasting verify. This is the predicted failure mode of the
shared-memory **same-position single-head reuse**: the head was not trained to unroll that
far. (p0.85/n12 = 226.4 here vs 226.3 in the prior session -> good reproducibility.)

### Cross-model: Qwen3.6-27B-MTP, replace curve (single-head, growing-KV positions)

```
p_min n_max  tg     acc    mean_len
0.80  4      140.2  0.996  4.84
0.80  8      185.1  0.981  7.83
0.80  12     215.9  0.966  10.20
0.80  16     225.2  0.945  11.65
0.80  24     227.0  0.912  12.91
0.85  4      137.0  0.997  4.82
0.85  8      182.9  0.987  7.73
0.85  12     210.7  0.975  9.99
0.85  16     222.7  0.956  11.39
0.85  24     223.9  0.944  12.39
```

Qwen's single-head, **growing-KV-position** path holds acceptance far better at depth:
n=24 keeps `acc = 0.91-0.94` and `mean_len ~= 12.5`, vs gemma's collapse to 0.70-0.72 at the
same depth. The curve is monotone and plateaus (~225 t/s) from n=16; no turnover through 24.
Baseline echo acceptance is near-perfect (0.997). The contrast isolates the mechanism:
**same-position reuse is gemma's deep-unroll bottleneck, not MTP drafting per se.**

### Cross-model takeaway

- Both models roughly **double** throughput vs the shipped `n_max = 4` in the echo regime
  (gemma 118 -> 234, Qwen 137 -> 224).
- **Optimal depth ceiling is architecture-dependent.** gemma (same-position) must be capped
  around 16 and *hard-collapses* past it; Qwen (growing-KV) is safe to 24+ and degrades
  gracefully. So an adaptive controller's *maximum* `n_max` should be per-model (or
  auto-discovered from the acceptance-vs-depth curve), not a shared constant.
- The `acc @ deep n` value is the architectural fingerprint: a head that keeps acceptance up
  at depth (Qwen) can be driven harder than one that falls off a cliff (gemma).

### Baseline-anchored speedups (replace, p0.80)

`n_max=0` crashes (see note), so the measured floor is `n_max=1` (minimal draft). Pure
no-spec throughput is estimated as `tg / mean_len` (the target forward-pass rate) - roughly
40 t/s for gemma from the n=1 row; it is a floor estimate, since n=1 still carries one draft
token of overhead.

gemma:

```
n_max  tg      mean_len   vs n=1    vs no-spec (~40)
1       80.1    1.99      1.00x     ~2.0x
4      118.7    4.78      1.48x     ~3.0x     <- shipped
8      189.1    8.12      2.36x     ~4.7x
12     203.9   10.07      2.54x     ~5.1x
16     214.3   12.03      2.68x     ~5.3x     <- peak
24     202.6   12.97      2.53x     ~5.0x
```

Qwen (no-spec est ~42 t/s from the n=1 row):

```
n_max  tg      mean_len   vs n=1    vs no-spec (~42)
1       83.3    2.00      1.00x     ~2.0x
4      140.2    4.84      1.68x     ~3.3x
8      185.1    7.83      2.22x     ~4.4x
12     215.9   10.20      2.59x     ~5.1x
16     225.2   11.65      2.70x     ~5.4x
24     227.0   12.91      2.73x     ~5.4x   <- plateau (still climbing, no collapse)
```

The reframing matters: the shipped `n=4` is only ~1.5-1.7x the minimal-draft floor (~3x
no-spec), while the echo peak is ~2.7x the floor (~5x no-spec). The `n=4 -> n=16` move alone
is ~1.6-1.8x. This is the number to quote, not the "+89% vs n=4" framing. Both models share
a near-identical no-spec floor (~40 t/s) and a ~2x floor at n=1; they diverge at depth -
gemma peaks then collapses, Qwen plateaus.

### Cap vs gate: the full 2x2 (tok/s), and its model-dependence

Measured no-spec floor (pure target, draft disabled): gemma 47, Qwen 50 (the earlier ~40/42
estimate from tg/mean_len at n=1 undercounted - n=1 carries draft overhead). So n=16 is ~4.5x
no-spec on both.

gemma (echo=replace / reasoning=refactor):

```
             p_min=0.0   p_min=0.8
replace  n4    121.7       118.2
replace  n16   187.5       218.4
refactor n4    107.6       112.2
refactor n16   143.2       163.5
```

Gate at n=16, both models x both content types (tok/s):

```
model  content     p=0.0   p=0.8   gate
gemma  recitation  187     218     +16%
gemma  reasoning   143     163     +14%
Qwen   recitation  227     225     ~0
Qwen   reasoning   140     178     +27%
```

Clean decomposition:
- Cap (n4 -> n16, gate off): +54%/+33% (gemma echo/reason), +60% (Qwen echo). The cap is the
  primary lever - beats n=4 on its own.
- Gate (p 0.0 -> 0.8) is **waste-dependent, not model-dependent**: it pays wherever the head
  drafts deep-but-wrong. On reasoning BOTH heads guess badly at depth (Qwen acc 0.40 -> 0.95,
  draft tokens 8862 -> 3727), so the gate is a big win on both (+14% to +27%). The lone neutral
  cell is Qwen on pure recitation - its growing-KV head is actually accurate that deep, so
  there's no tail to prune. At n=4 the gate is neutral on both (no deep tail yet). So on
  realistic mixed traffic the gate helps both models; the earlier "neutral on Qwen" was an
  echo-only artifact. Still not a global default (inert at the shipped n_max=3, and a global
  param governing draft-model/n-gram paths too). This is also why per-token gating beats the
  round-level controller.

## Recommendation

1. **Static bump first (no code):** `n_max = 12`, `p_min ~= 0.80-0.85`. Robust +42% to +89%
   over the shipped `n_max = 4` across all three tasks; even the most reasoning-heavy task
   (refactor) gains ~40%. The confidence gate (`p_min`) is what makes a deep cap safe — it
   self-truncates per token when the head loses confidence, so raising the cap costs little
   on low-`a` stretches.

2. **Adaptive depth (the actual goal):** honor `dp.n_max` in the MTP draft loop (it is
   already plumbed from the server but currently ignored), keep `params.n_max` as the
   ceiling, and set `dp.n_max` per round from an AIMD / EWMA of recent acceptance in the
   server (increase on `a` high, multiplicative-decrease on `a` low, with hysteresis). The
   response curve above parameterizes it: `a ~ 0.97 -> n ~ 16+`, `a ~ 0.90 -> n ~ 12`,
   `a ~ 0.85 -> n ~ 8`.

   **Hard design constraint (from the n_max=0 crash below):** the output buffer is sized
   `n_outputs_per_seq = 1 + common_speculative_n_max(...)` **once at server init**
   (`tools/server/server-context.cpp:50`), from the *configured* `--spec-draft-n-max`. So
   the controller must set the configured `n_max` to the intended **ceiling** (buffer sized
   for the deepest draft) and only ever throttle `dp.n_max` **down** from there per round -
   which matches the existing `min`-clamp semantics of `dp.n_max` in the other impls. It
   must also never drive effective depth to 0 (floor at 1).

### Note: n_max=0 aborts

`--spec-draft-n-max 0` crashes draft-mtp:
`GGML_ASSERT(n_outputs_max <= cparams.n_outputs_max)` in `llama_context::output_reserve`
(`src/llama-context.cpp:2219`). The output buffer is reserved for `1 + 0 = 1` output, but
the draft path still runs and drafts 1 token, so the verify decode needs 2 output slots.
Consequence: n_max=0 is not a valid "framework on, drafting off" baseline; the minimal
working floor is `n_max=1`. A one-line upstream guard (skip drafting when the effective cap
is 0) would make n_max=0 a clean no-op baseline.

## Controller: built, validated, and shelved (negative result)

Implemented an opt-in (`LLAMA_SPEC_ADAPTIVE_DEPTH`) round-level depth controller: the MTP
`draft()` loop honors a per-round `dp.n_max` cap (`common/speculative.cpp`), and the server
drives it. The configured `--spec-draft-n-max` stays the ceiling (buffer sizing); the
controller only throttles under it.

Two control laws tried:
1. Aim one past `EWMA(accepted_run)` - **death-spiralled**: the measured run is capped by the
   current depth, so shrinking depth shrinks the signal and it collapses to depth ~5-9. Was
   worse than fixed on every task.
2. Throughput hill-climb (extremum seeking: sample tok/s over a window, step depth +/-1,
   reverse on regression) - **stable, achieves parity**, but no win. A pure acceptance signal
   provably can't locate the peak (it sits at acc ~0.83 for echo but ~0.71 for reasoning), so
   optimizing measured throughput is the only content-agnostic option; it just finds that the
   peak is already where a fixed cap sits.

Validation (gemma, p0.80, tok/s):

```
                       fixed      adaptive
parity  replace c=16   218        222
parity  refactor c=16  164        164
robust  replace c=24   213        209     (curve is a flat, noisy plateau - no collapse to rescue)
bimodal reason->recite 117        114     (fixed-16 beats fixed-4's 95; adaptive ~parity)
```

Verdict: **the controller does no harm but earns no win.** Two confirmed reasons: (a) `p_min`
already adapts per-token, finer than any round-level cap - in the bimodal test the reasoning
half is truncated by `p_min`, so a high fixed cap is not penalized there; (b) the
depth-throughput landscape is a broad plateau (gemma n=12..24 all ~205-218), so precise depth
control buys little over picking a reasonable fixed `n_max`. Ship fixed `n_max`+`p_min`; the
gated controller stays as documented scaffolding only.

## Open questions

- ~~Where is the echo-regime ceiling?~~ **Answered:** gemma peaks at n~=16 and collapses by
  n=24; Qwen plateaus from n~=16, no collapse through 24. Ceiling is per-architecture.
- ~~Does the single-head Qwen path unroll deeper than gemma's same-position reuse?~~
  **Answered: yes, clearly** — Qwen holds acc 0.91+ at n=24 where gemma drops to 0.70.
- Real agent streams interleave regimes within one generation; the sweep uses homogeneous
  tasks. Need a mixed prompt (reason-then-recite) to confirm the controller tracks the
  switch, not just the steady state.
- Does the collapse point move with the target/draft quant, context length, or temperature?
  The ceiling was measured at one operating point per model.
