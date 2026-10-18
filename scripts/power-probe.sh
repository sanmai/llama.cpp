#!/usr/bin/env bash
#
# power-probe.sh - per-phase GPU power + energy/token for a model.
#
# Separates prefill from decode into their own runs so power is never averaged
# across phases; derives J/tok = steady-state active power / throughput (no
# integration: over a sustained run the GPU sits at stable power and throughput).
# "active" = util > 90%; idle baseline (GPU at rest, sampled before model load)
# gives the marginal-energy column. Run on an otherwise-idle GPU.
#
# Prefill defaults to REAL text (benchmark.bin via llama-completion) rather than
# synthetic repeated tokens - a representative token distribution / attention span.
#
# Usage: ./scripts/power-probe.sh MODEL [phase ...]
#   phases:
#     file[:PATH]  real-text prefill via llama-completion on PATH (default $PROMPT_FILE)
#     ppN          synthetic prefill of N tokens (llama-bench)
#     tgN          decode N tokens (llama-bench)
#   default: "file tg256" if $PROMPT_FILE exists, else "pp4096 tg256"
#   env: BIN, COMPLETION, REPS, BATCH, UBATCH, NGL, PROMPT_FILE, CTX, CACHE_TYPE
#
# Example: ./scripts/power-probe.sh qwen3.6-27b-q4_0.gguf
#          ./scripts/power-probe.sh qwen3.6-27b-n4_0.gguf file tg256

set -euo pipefail

# force dot decimals: this locale prints "3599,76" which breaks numeric parsing
export LC_ALL=C

BIN="${BIN:-build-local/bin/llama-bench}"
COMPLETION="${COMPLETION:-build-local/bin/llama-completion}"
REPS="${REPS:-20}"
BATCH="${BATCH:-8192}"
UBATCH="${UBATCH:-4096}"
NGL="${NGL:-999}"
PROMPT_FILE="${PROMPT_FILE:-benchmark.bin}"
CTX="${CTX:-81920}"
CACHE_TYPE="${CACHE_TYPE:-q8_0}"

if [ $# -lt 1 ]; then
    echo "usage: $0 MODEL [phase ...]" >&2
    exit 1
fi
MODEL="$1"; shift
phases=("$@")
if [ ${#phases[@]} -eq 0 ]; then
    if [ -f "$PROMPT_FILE" ]; then
        phases=(file tg256)
    else
        phases=(pp4096 tg256)
    fi
fi

for b in "$BIN" "$COMPLETION"; do
    [ -x "$b" ] || { echo "missing $b - build first (./build-local.sh)" >&2; exit 1; }
done
command -v jq >/dev/null || { echo "jq required" >&2; exit 1; }

sample() {
    nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv,noheader,nounits -lms 100
}

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

# true idle baseline (GPU at rest, before any model load) for marginal energy.
# clocks stay boosted ~10 s after a heavy run, so wait out the decay before sampling.
sleep 120
: > "$LOG"
sample >> "$LOG" &
idle_pid=$!
sleep 1.5
kill "$idle_pid" 2>/dev/null || true
wait "$idle_pid" 2>/dev/null || true
# min over low-util samples - robust to any residual decay tail
IDLE_W=$(awk -F', *' '$2<10 {if (!n || $1<m) m=$1; n++} END {printf "%.1f", (n ? m : 0)}' "$LOG")
echo "idle baseline: ${IDLE_W} W"

printf '%-12s %12s %10s %10s %12s\n' "PHASE" "t/s" "ACTIVE_W" "J/tok" "MARG_J/tok"
for ph in "${phases[@]}"; do
    : > "$LOG"
    sample >> "$LOG" &
    logger_pid=$!

    case "$ph" in
        file*)
            pf="${ph#file:}"
            [ "$pf" = "file" ] && pf="$PROMPT_FILE"
            [ -f "$pf" ] || { echo "missing prompt file '$pf'" >&2; exit 1; }
            out=$(timeout --foreground -k 30 600 \
                "$COMPLETION" -m "$MODEL" -f "$pf" -n 1 -fa on -ngl "$NGL" \
                -b "$BATCH" -ub "$UBATCH" -c "$CTX" -ctk "$CACHE_TYPE" -ctv "$CACHE_TYPE" \
                -no-cnv --no-display-prompt --no-warmup --simple-io --temp 0 --top-k 1 2>&1) || true
            tps=$(printf '%s\n' "$out" | sed -n 's/.*prompt eval time.*, *\([0-9.]*\) tokens per second.*/\1/p' | tail -1)
            ;;
        pp*|tg*)
            case "$ph" in
                pp*) args=(-p "${ph#pp}" -n 0 -b "$BATCH" -ub "$UBATCH") ;;
                tg*) args=(-p 0 -n "${ph#tg}") ;;
            esac
            json=$("$BIN" -m "$MODEL" "${args[@]}" -r "$REPS" -ngl "$NGL" -fa 1 -o json 2>/dev/null) || true
            tps=$(printf '%s' "$json" | jq -r '.[0].avg_ts // empty')
            ;;
        *) echo "bad phase '$ph' (want file[:PATH], ppN, or tgN)" >&2; exit 1 ;;
    esac

    kill "$logger_pid" 2>/dev/null || true
    wait "$logger_pid" 2>/dev/null || true

    [ -z "${tps:-}" ] && tps=0
    active_w=$(awk -F', *' '$2>90 {a+=$1; na++} END {printf "%.1f", (na ? a/na : 0)}' "$LOG")

    awk -v ph="$ph" -v tps="$tps" -v aw="$active_w" -v iw="$IDLE_W" 'BEGIN {
        jtok = (tps > 0 ? aw / tps : 0);
        marg = (tps > 0 ? (aw - iw) / tps : 0);
        printf "%-12s %12.2f %10.1f %10.3f %12.3f\n", ph, tps, aw, jtok, marg;
    }'
done
