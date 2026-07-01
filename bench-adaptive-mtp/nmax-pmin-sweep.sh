#!/bin/bash
# 2D sweep of MTP draft depth (n_max) x confidence gate (p_min) for gemma-4-31B.
#
# Usage: ./tmp/nmax-pmin-sweep.sh [task] [from] [to]
#   task = refactor | replace | autotype   (default: refactor)
#   from/to = identifier pair for the mechanical `replace` task (default: float/float32).
#             Pick a rare identifier for high acceptance, a common one for low -> this is
#             the "density knob" that dials alpha as an independent variable.
#
# All three tasks recite the SAME draft() method, so recitation content is held constant
# and only the edit type varies:
#   refactor  - structural edit (extract a helper); realistic clustered edits.
#   replace   - mechanical substitution; alpha dialled by `from` frequency.
#   autotype  - replace `auto` with the deduced concrete type; edits require reasoning.
#
# The speculative context (frozen MTP impl params) is built once at server init, so
# per-request overrides never reach the draft-mtp impl -> we restart per (p_min, n_max)
# config. We reuse the known-good launcher and override only the swept knobs via its
# trailing "$@" (CLI wins over the env vars it hardcodes, arg.cpp:650).
#
# Metrics are scraped from the guaranteed server log INFO lines:
#   server-context.cpp:576  "n_decoded = .., tg = .. t/s, tg_3s = .. t/s"
#   server-context.cpp:629  "draft acceptance = .. (a accepted / g generated), mean len = .."
set -u

TASK="${1:-refactor}"
FROM="${2:-float}"
TO="${3:-float32}"

PORT=8099
LAUNCHER="${LAUNCHER:-./gemma-4-31B-it-GGUF.sh}"   # override to sweep a different model
TAG=$(basename "$LAUNCHER" .sh)
CTXFILE="common/speculative.cpp"
CTX=40960
NPRED=4096   # let it rip: long responses sustain the recitation regime and average out noise
SEED=1234
RESULTS="tmp/sweep-${TASK}-${TAG}-results.tsv"

# grid groups "p_min:n_max,n_max,..." separated by spaces; override via GRID_SPEC env
GRID_DEFAULT="0.75:4,5,6,8 0.80:4,6,8,12 0.85:6,8,12 0.90:8,12"
read -ra GRID <<< "${GRID_SPEC:-$GRID_DEFAULT}"

mkdir -p tmp

# All tasks target the whole struct common_speculative_impl_draft_mtp -> long response,
# constant recitation content, plenty of `float`/`auto` occurrences for the edit tasks.
case "$TASK" in
    refactor)
        QUESTION="Below is the complete file common/speculative.cpp. Refactor struct common_speculative_impl_draft_mtp so the body of the per-sequence inner while-loop step inside draft() is extracted into a new private helper method named draft_step(). Keep behaviour identical. Emit the complete refactored struct verbatim." ;;
    replace)
        QUESTION="Below is the complete file common/speculative.cpp. Reproduce struct common_speculative_impl_draft_mtp in full, verbatim, but replace every occurrence of the identifier \`${FROM}\` with \`${TO}\`. Output only the modified struct, nothing else." ;;
    autotype)
        QUESTION="Below is the complete file common/speculative.cpp. Reproduce struct common_speculative_impl_draft_mtp in full, verbatim, but replace every \`auto\` type specifier with the concrete C++ type it deduces to. Always commit to a concrete type: if you cannot determine it with certainty, write your single best guess rather than leaving \`auto\`. Output only the modified struct, nothing else." ;;
    mixed)
        QUESTION="Below is the complete file common/speculative.cpp. Reproduce struct common_speculative_impl_draft_mtp in full, verbatim, but immediately before each method definition insert one new one-line comment (starting with //) explaining in your own words what that method does. Keep every method body verbatim. Output only the modified struct, nothing else." ;;
    bimodal)
        QUESTION="Below is the complete file common/speculative.cpp. Do two things in order. First, write an original analysis of at least 500 words (prose only, no code, do not quote the file) of the concurrency and correctness hazards in the draft() method of struct common_speculative_impl_draft_mtp. Second, after a line containing only ---, reproduce that draft() method verbatim, character for character." ;;
    *)
        echo "unknown task: $TASK (want refactor|replace|autotype)"; exit 1 ;;
esac

echo "task=$TASK  launcher=$LAUNCHER  from=$FROM  to=$TO  grid='${GRID[*]}'  -> $RESULTS"

# Build the chat request body once; same prompt for every config.
BODY=$(jq -n \
    --rawfile f "$CTXFILE" \
    --arg q "$QUESTION" \
    --argjson np "$NPRED" \
    --argjson seed "$SEED" \
    '{messages:[{role:"user",content:($q + "\n\n--- common/speculative.cpp ---\n" + $f)}],
      n_predict:$np, temperature:0.6, top_p:0.95, top_k:20,
      seed:$seed, cache_prompt:true, stream:false}')

req() {
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$BODY"
}

# Launcher bash and llama-server land in different process groups, so tear down by the
# unique port match (SIGTERM, then SIGKILL) and wait for VRAM to free.
cleanup_server() {
    pkill -TERM -f "llama-server.*--port ${PORT}" 2>/dev/null
    for _ in $(seq 1 20); do
        curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1 || return 0
        sleep 1
    done
    pkill -KILL -f "llama-server.*--port ${PORT}" 2>/dev/null
    sleep 3
}

printf 'p_min\tn_max\ttg_t/s\tn_dec\tacc\tgen\tmean_len\n' | tee "$RESULTS"

for spec in "${GRID[@]}"; do
    P="${spec%%:*}"
    IFS=',' read -ra NMAX <<< "${spec#*:}"

    for N in "${NMAX[@]}"; do
        LOG="tmp/sweep-${TASK}-${TAG}-p${P}-n${N}.log"

        cleanup_server   # clear any straggler before launching

        "$LAUNCHER" \
            --spec-draft-n-max "$N" --spec-draft-p-min "$P" \
            -c "$CTX" --parallel 1 --port "$PORT" --no-webui > "$LOG" 2>&1 &
        SRV=$!

        ok=0
        for _ in $(seq 1 300); do
            sleep 1
            if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then ok=1; break; fi
            kill -0 "$SRV" 2>/dev/null || break
        done

        if [ "$ok" -ne 1 ]; then
            printf '%s\t%s\tSTART_FAIL\t-\t-\t-\t-\n' "$P" "$N" | tee -a "$RESULTS"
            cleanup_server; wait "$SRV" 2>/dev/null
            continue
        fi

        req >/dev/null 2>&1   # warmup: fill prompt cache + graph
        req >/dev/null 2>&1   # measured request (metrics scraped from log tail)

        tg_line=$(grep 'tg = '              "$LOG" 2>/dev/null | tail -1)
        acc_line=$(grep 'draft acceptance = ' "$LOG" 2>/dev/null | tail -1)

        tg=$(printf  '%s' "$tg_line"  | sed -nE 's/.*tg = *([0-9.]+) t\/s.*/\1/p')
        nd=$(printf  '%s' "$tg_line"  | sed -nE 's/.*n_decoded = *([0-9]+),.*/\1/p')
        acc=$(printf '%s' "$acc_line" | sed -nE 's/.*draft acceptance = *([0-9.]+) .*/\1/p')
        gen=$(printf '%s' "$acc_line" | sed -nE 's#.*/ *([0-9]+) generated.*#\1#p')
        ml=$(printf  '%s' "$acc_line" | sed -nE 's/.*mean len = *([0-9.]+).*/\1/p')

        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$P" "$N" "${tg:-?}" "${nd:-?}" "${acc:-?}" "${gen:-?}" "${ml:-?}" | tee -a "$RESULTS"

        cleanup_server; wait "$SRV" 2>/dev/null
    done
done

echo "done ($TASK) -> $RESULTS"
