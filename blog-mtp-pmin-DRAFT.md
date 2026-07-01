# Doubling MTP speculative decoding throughput: raise the draft cap, then gate it

*Draft — rewrite in your own voice before publishing.*

## TL;DR

Two flags — `--spec-draft-n-max 16 --spec-draft-p-min 0.8` — give **~3.5x** the throughput of
plain decoding on a realistic code-editing task (refactor a function and emit the result),
measured on two different MTP models (tok/s):

| config | gemma-4-31B | Qwen3.6-27B |
|---|---:|---:|
| no speculation | 47 | 50 |
| `n_max=1` | 78 | 77 |
| `n_max=4` (≈ shipped default) | 112 | 112 |
| **`n_max=16, p_min=0.8`** | **163** | **178** |

Most of the win is raising the draft cap; the confidence gate (`p_min`) adds up to +27% on
this kind of editing work and is free where it doesn't help. On pure recitation (reproducing
unchanged code) it goes higher still — ~4.5x. Why, below.

---

llama.cpp ships MTP (multi-token-prediction) speculative decoding: a small extra head on the
model proposes several tokens ahead, the full model verifies them in a single forward pass,
and every token the model would have produced anyway comes for free. The knob that governs how
far ahead it drafts, `--spec-draft-n-max`, defaults to 3.

On a 5090 I measured two MTP models — gemma-4-31B (Q4_K_XL) and Qwen3.6-27B (Q6_K) — and got
up to ~2x over the shipped default and ~4.5x over no speculation by raising that cap, plus a
second flag whose job is to keep the deep cap honest. The interesting part is *when* the second
flag pays off: not per-model, but per-content — it helps wherever the head drafts deep-but-wrong,
which turns out to be most real work on both models.

## The two flags

```
--spec-draft-n-max 16      # draft up to 16 tokens per step (default 3)
--spec-draft-p-min 0.8     # stop drafting once the head's confidence drops below 0.8 (default 0)
```

Throughput on a recitation-heavy prompt (tok/s):

| config | gemma-4-31B | Qwen3.6-27B |
|---|---:|---:|
| no speculation | 47 | 50 |
| `n_max=1` | 80 | 83 |
| `n_max=4` | 118 | 140 |
| **`n_max=16, p_min=0.8`** | **218** | **225** |

Roughly 4.5x over pure decoding on both models, almost all of it from raising the cap.

## Why a deeper draft helps

Acceptance comes in bursts. When the model reproduces something it can predict well — quoting
a file, emitting the unchanged context lines of a diff, reciting a function it just read — the
head is almost always right, and a long run of proposed tokens is accepted in a single
verification. Coding-agent traffic is dominated by this kind of echo, so the head can often
ride 10+ tokens per step. A cap of 3 throws most of that away; raising it to ~16 captures the
runs. Past that you hit diminishing returns, and on some heads a collapse (below).

## What the confidence gate does

Not all output is echo. In the reasoning stretches — deciding *what* to write, novel logic,
prose — the head is guessing, and its deep proposals get rejected. `p_min` stops the draft the
moment the head's own confidence drops below the threshold, so a deep cap only spends depth
where the head is sure.

First, the cap does the heavy lifting on its own — `n_max` 4→16 at `p_min=0`:

| | gemma | Qwen |
|---|---:|---:|
| `n_max=4`, `p_min=0` | 122 | 142 |
| `n_max=16`, `p_min=0` | 187 | 227 |

Roughly +55% / +60%, gate or no gate. Now turn the gate on at `n_max=16`, across both models
and both kinds of content:

| model | content | p_min=0.0 | p_min=0.8 | gate |
|---|---|---:|---:|---:|
| gemma | recitation | 187 | 218 | +16% |
| gemma | refactor | 143 | 163 | +14% |
| Qwen | recitation | 227 | 225 | ~0 |
| Qwen | refactor | 140 | 178 | **+27%** |

The gate pays **wherever the head drafts deep-but-wrong**. On the refactor both heads guess
badly at depth (Qwen's acceptance craters to 0.40 without the gate, and it drafts more than
twice as many tokens as it keeps), so pruning the confident-but-wrong tail is a large win —
+14% to +27%. The lone exception is Qwen on pure recitation: its head advances KV positions and stays
accurate that deep, so there's no wasted tail to cut. The gate isn't model-dependent so much
as *waste-dependent*, and real mixed traffic has plenty of waste to cut on both models.

One thing it is *not*: a shallow-cap win. At the shipped `n_max=3` there's no deep tail to
prune, and the gate is within noise of nothing. Raise the cap first; add the gate to keep the
depth honest.

## The twist: we tried to be clever and lost

The obvious next move is an adaptive controller — watch acceptance, grow the draft during echo
bursts, shrink it during reasoning. I built one (a throughput hill-climb over depth) and
benchmarked it against a plain fixed cap on four workloads: recitation, refactoring,
comment-writing, and a deliberately bimodal reason-then-recite prompt.

It never won — parity at best. The reason is the same `p_min`: a confidence gate already
adapts the depth *per token*, which is finer than any once-per-verification controller. On the
bimodal prompt the reasoning half is truncated by the gate, so a high fixed cap isn't
penalized there in the first place — there's no gap left for a round-level controller to
exploit. The simple knob had already eaten its lunch.

## What to actually do

For MTP speculative decoding on echo-heavy (agentic, code) traffic:

```
--spec-draft-n-max 16 --spec-draft-p-min 0.8
```

Set the cap to your head's pre-collapse ceiling — measure it: gemma's head peaks near 16 and
degrades past it, Qwen's plateaus out to 24 — and turn the gate on. It helps on almost
everything (+14% to +27% here) and is free on the rest. Don't reach for an adaptive
controller; the confidence gate is the adaptation.

And don't just flip the `p_min` default: it does nothing at the shipped cap of 3 (no deep tail
to prune — you have to raise `n_max` first), and it's a global speculative parameter that also
governs draft-model and n-gram paths this was never measured on. It's a tuning recommendation
paired with a deep cap, not a standalone default.

## Caveats

Two MTP model families, one quant each, one temperature; throughput is workload-dependent
(recitation wins most, refactoring least). The best `p_min` value and the depth ceiling are
head-specific — sweep them. The "~2x" is against llama.cpp's default `n_max=3`; the "~4.5x" is
against no speculation at all. Numbers are single-shot; treat sub-10% gaps as noise.
