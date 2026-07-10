#!/usr/bin/env bash
#
# nvfp4-weight-kld.sh - weight-quant KLD harness for the N4_K -> N4_0 walk-back.
#
# For each gguf, runs llama-perplexity --kl-divergence on the native FP4 path
# (-ngl 999) against the f16 base and tabulates Mean KLD + Same-top-p. References
# (q4_0, q3_k) go through the identical command for an apples-to-apples column.
# Every arm uses the same --chunks count for internal comparability.
#
# Usage: ./scripts/nvfp4-weight-kld.sh [gguf ...]
#   No args: runs the 27B walk-back set + q4_0/q3_k references in CWD
#   (missing files print MISSING so the table fills in as arms are built).

set -euo pipefail

BIN="${BIN:-build-local/bin/llama-perplexity}"
BASE="${BASE:-kld-base/qwen3.6-27b-f16-kld-base.dat}"
DATA="${DATA:-wikitext-2-raw/wiki.test.raw}"
CHUNKS="${CHUNKS:-200}"

ARGS=(
    -f "${DATA}"
    --kl-divergence
    --kl-divergence-base "${BASE}"
    --chunks "${CHUNKS}"
    -ngl 999
    -c 512
    -b 4096
    -ub 2048
    -fa on
)

if [ ! -x "${BIN}" ]; then
    echo "missing ${BIN} - build first (./build-local.sh)" >&2
    exit 1
fi
if [ ! -f "${BASE}" ]; then
    echo "missing KLD base ${BASE} - generate it first" >&2
    exit 1
fi

models=("$@")
if [ ${#models[@]} -eq 0 ]; then
    models=(
        qwen3.6-27b-n4k-pow2.gguf
        qwen3.6-27b-n4_0-search.gguf
        qwen3.6-27b-n4_0.gguf
        qwen3.6-27b-q4_0.gguf
        qwen3.6-27b-q3_k.gguf
    )
fi

printf '%-32s %16s %14s %8s\n' "MODEL" "MEAN_KLD" "SAME_TOP_P%" "SIZE"
for m in "${models[@]}"; do
    if [ ! -f "$m" ]; then
        printf '%-32s %16s\n' "$m" "MISSING"
        continue
    fi

    out=$("${BIN}" -m "$m" "${ARGS[@]}" 2>/dev/null) || true

    kld=$(printf '%s\n' "$out" | sed -n 's/^Mean[[:space:]]*KLD:[[:space:]]*\([0-9.]*\).*/\1/p' | tail -1)
    top=$(printf '%s\n' "$out" | sed -n 's/^Same top p:[[:space:]]*\([0-9.]*\).*/\1/p' | tail -1)
    sz=$(du -h --apparent-size "$m" | cut -f1)

    printf '%-32s %16s %14s %8s\n' "$m" "${kld:-ERR}" "${top:-ERR}" "$sz"
done
