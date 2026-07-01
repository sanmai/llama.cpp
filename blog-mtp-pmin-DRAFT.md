# Doubling MTP speculative decoding throughput with one confidence knob

*Draft — rewrite in your own voice before publishing.*

llama.cpp ships MTP (multi-token-prediction) speculative decoding: a small extra head on
the model proposes several tokens, the full model verifies them in a single forward pass, and
every proposed token the model would have produced anyway comes for free. The knob that
governs it, `--spec-draft-n-max`, defaults to 3. On a 5090 with gemma-4-31B I got roughly
**2x more tokens/second** by changing two flags — and the interesting part is *why* the second
flag is what makes the first one safe.

## The two flags

```
--spec-draft-n-max 16      # draft up to 16 tokens per step (default 3)
--spec-draft-p-min 0.8     # stop drafting once the head's confidence drops below 0.8 (default 0)
```

Measured throughput (gemma-4-31B, Q4_K_XL, recitation-heavy prompt, tokens/s):

```
no speculation (est.)   ~40
n_max=1  (floor)          80
n_max=4                  118
n_max=16, p_min=0.8      218      <- ~2x over the default, ~5x over no speculation
```

## Why a deeper draft helps

Acceptance is not uniform — it comes in bursts. When the model is reproducing something it can
predict well (quoting a file, emitting the unchanged lines of a diff, reciting a function it
just read), the head is almost always right, and long runs of proposed tokens get accepted in
a single verification. Coding-agent output is dominated by exactly this kind of echo, so the
draft head can often ride 10+ tokens per step. A cap of 3 leaves most of that on the table;
raising it to ~16 captures the long runs. Push it much further and you hit diminishing returns
(and, on some heads, a collapse — see below).

## Why the confidence gate is what makes it safe

Here's the catch: not all output is echoey. In the reasoning stretches — deciding *what* to
change, writing novel logic, prose — the head is guessing, and deep proposals just get
rejected. Drafting 16 tokens there burns compute (the extra draft passes and a wider
verification batch) for tokens the model throws away.

`p_min` fixes this per token: the head stops proposing the moment its own confidence drops
below the threshold. So a high cap costs almost nothing on low-confidence content — it only
spends depth where the head is sure. The ablation, both at `n_max=16`:

```
                p_min=0.0    p_min=0.8
echo task        187 t/s      218 t/s     draft tokens 6113 -> 4529, acceptance 0.61 -> 0.83
reasoning task   143 t/s      163 t/s     draft tokens 8251 -> 5044, acceptance 0.43 -> 0.71
```

Turning the gate on prunes 30–60% of the proposed tokens — precisely the low-confidence tail
that was going to be rejected — and gains ~15% throughput while keeping the same accepted run
length. The deep cap does the heavy lifting; the gate keeps it from wasting effort.

## The twist: we tried to be clever and lost

The obvious next move is an adaptive controller: watch the acceptance rate and grow the draft
depth during echo bursts, shrink it during reasoning. I built one (a throughput hill-climb
over depth) and benchmarked it against the plain fixed cap on four workloads — pure echo,
pure reasoning, annotated code, and a deliberately bimodal reason-then-recite prompt.

It never won. It reached parity and no more. The reason is the same `p_min`: a confidence gate
already adapts the depth *per token*, which is finer than any once-per-verification-round
controller can manage. On the bimodal prompt, the reasoning half is truncated by the gate, so
a high fixed cap isn't penalized there in the first place — there's no gap for a round-level
controller to exploit. The simple knob had already eaten the controller's lunch.

## What to actually do

For MTP speculative decoding on echo-heavy (agentic, code) workloads:

```
--spec-draft-n-max 16 --spec-draft-p-min 0.8
```

Set the cap to your head's pre-collapse ceiling (measure it — one head plateaued out to 24,
another peaked at 16 and degraded past it) and turn the gate on. Don't reach for an adaptive
controller; the confidence gate is the adaptation.

## Caveats

Single model family, one quant, one temperature; throughput is workload-dependent (echo-heavy
wins most, pure reasoning least). `p_min`'s best value and the depth ceiling are
model-specific — sweep them. The absolute "2x" is against llama.cpp's default `n_max=3`; your
mileage depends on how echoey your traffic is.
