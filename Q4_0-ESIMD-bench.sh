#!/usr/bin/env bash
# Q4_0 ESIMD A/B re-bench + correctness check. Toggles GGML_SYCL_DEQUANT_ESIMD
# between 0 and 1 on a single build, so any delta is attributable to the kernel
# swap alone. Run after refactoring the ESIMD kernel/launcher to confirm no
# perf or PPL regression vs the parallel_for baseline.
#
#   MODEL=...       override Q4_0 model path (default: qwen2.5-7b-q4_0.gguf)
#   WIKITEXT=...    override perplexity input (default: wikitext-2-raw/wiki.test.raw)
#   REPS=N          override bench -r (default: 5)
#   CHUNKS=N        override perplexity --chunks (default: 4; bump to 100 for high confidence)
#   SKIP_BENCH=1    skip the perf bench
#   SKIP_PPL=1      skip the correctness check

set -euo pipefail

MODEL=${MODEL:-qwen2.5-7b-q4_0.gguf}
WIKITEXT=${WIKITEXT:-wikitext-2-raw/wiki.test.raw}
REPS=${REPS:-5}
CHUNKS=${CHUNKS:-4}
BENCH=./build-local/bin/llama-bench
PERPLEXITY=./build-local/bin/llama-perplexity

if [[ ! -x $BENCH ]] || [[ ! -x $PERPLEXITY ]]; then
    echo "error: build-local binaries missing - run ./build-local.sh first" >&2
    exit 1
fi
if [[ ! -e $MODEL ]]; then
    echo "error: $MODEL not found - set MODEL=... or symlink the Q4_0 model" >&2
    exit 1
fi

commit=$(git rev-parse --short HEAD)
echo "=== Q4_0 ESIMD A/B @ $commit ==="
echo "model:    $MODEL"
echo "reps:     $REPS"
echo "chunks:   $CHUNKS"
echo

if [[ -z ${SKIP_BENCH:-} ]]; then
    echo "--- bench: ESIMD=0 (parallel_for baseline) ---"
    GGML_SYCL_DEQUANT_ESIMD=0 "$BENCH" -m "$MODEL" -p 4096 -n 128 -r "$REPS" -b 4096 -ub 2048
    echo
    echo "--- bench: ESIMD=1 (this PR) ---"
    GGML_SYCL_DEQUANT_ESIMD=1 "$BENCH" -m "$MODEL" -p 4096 -n 128 -r "$REPS" -b 4096 -ub 2048
    echo
fi

if [[ -z ${SKIP_PPL:-} ]]; then
    if [[ ! -e $WIKITEXT ]]; then
        echo "error: $WIKITEXT not found - set WIKITEXT=... or skip with SKIP_PPL=1" >&2
        exit 1
    fi
    echo "--- ppl: ESIMD=0 (baseline) ---"
    GGML_SYCL_DEQUANT_ESIMD=0 "$PERPLEXITY" -m "$MODEL" -f "$WIKITEXT" --chunks "$CHUNKS" | tail -2
    echo
    echo "--- ppl: ESIMD=1 (this PR) ---"
    GGML_SYCL_DEQUANT_ESIMD=1 "$PERPLEXITY" -m "$MODEL" -f "$WIKITEXT" --chunks "$CHUNKS" | tail -2
    echo
    echo "acceptance: |delta PPL| <= 0.01 for $CHUNKS chunks (reference: 6.7016 vs 6.7018 at 92dc04912)"
fi
