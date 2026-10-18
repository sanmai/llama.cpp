#!/usr/bin/env bash
#
# nvfp4-smoke.sh - fast coherence gate for quantized ggufs before trusting KLD.
#
# Greedy-decodes a few tokens from a strongly patterned prompt ("passed\n" x7).
# A SOUND model echoes the pattern; a cooked/garbage model emits gibberish. This
# is a coherence smoke test, not a quality measurement - it only separates
# "produces coherent text" from "broken".
#
# Usage: ./nvfp4-smoke.sh [gguf ...]
#   Defaults to qwen3.6-27b-q4_0.gguf and qwen3.6-27b-nvfp4*.gguf in CWD.
#
# NOTE: qwen3.6-27b-nvfp4-nosubn.gguf is a MEASUREMENT artifact - its UE4M3 scale
# bytes are lifted x64 with no in-file compensation, so it only reconstructs on a
# matching x64-encode/divide-64-decode build. On a stock binary it is EXPECTED to
# read GARBAGE here; that is not a sign the quantization itself is bad.

set -euo pipefail

BIN=build-local/bin/llama-completion
NEEDLE=passed

if [ ! -x "$BIN" ]; then
    echo "missing $BIN - build first (./build-local.sh)" >&2
    exit 1
fi

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
    shopt -s nullglob
    models=(qwen3.6-27b-q4_0.gguf qwen3.6-27b-n*.gguf qwen3.6-35b-a3b-n*.gguf  qwen3.6-35b-a3b-q*.gguf)
    shopt -u nullglob
fi

printf '%-42s %s\n' "MODEL" "VERDICT"
for m in "${models[@]}"; do
    if [ ! -f "$m" ]; then
        printf '%-42s %s\n' "$m" "MISSING"
        continue
    fi

    out=$(timeout --foreground -v -k 120 90 \
        "$BIN" -m "$m" -c 4096 -n 8 -fa on -ngl 999 \
        -ctk q8_0 -ctv q8_0 -no-cnv --no-display-prompt --no-warmup \
        --simple-io --temp 0 --top-k 1 \
        -p $'passed\npassed\npassed\npassed\npassed\npassed\npassed\n' 2>/dev/null) || true

    if printf '%s' "$out" | grep -q "$NEEDLE"; then
        printf '%-42s %s\n' "$m" "SOUND"
    else
        printf '%-42s %s\n' "$m" "GARBAGE"
    fi
done
