#!/usr/bin/env bash
set -euo pipefail
set -x

PPL_BIN="${PPL_BIN:-./llama-perplexity}"
MODEL="${MODEL:-OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf}"
DATA="${DATA:-wikitext-2-raw/wiki.test.raw}"
BASE="${BASE:-kld-base.dat}"
QUIET_CHUNKS="${QUIET_CHUNKS:-1}"

COMMON_ARGS=(
    -m "${MODEL}"
    -f "${DATA}"
    -ngl 999
    -c 512
    -b 4096
    -ub 2048
    -fa on
)

run_ppl() {
    if [[ "${QUIET_CHUNKS}" == "1" ]]; then
        "$@" | awk '
            /^chunk[[:space:]]+PPL[[:space:]]+/ { next }
            /^[[:space:]]*[0-9]+[[:space:]]+[0-9.]+[[:space:]]/ { next }
            { print }
        '
    else
        "$@"
    fi
}

run_compare() {
    local type_k="$1"
    local type_v="$2"

    echo "## llama-perplexity -ctk ${type_k} -ctv ${type_v}"
    echo

    MODE=compare run_ppl "${PPL_BIN}" \
        "${COMMON_ARGS[@]}" \
        --kl-divergence-base "${BASE}" \
        --kl-divergence \
        -ctk "${type_k}" \
        -ctv "${type_v}"
}

if [[ ! -f "${BASE}" ]]; then
    MODE=base run_ppl "${PPL_BIN}" \
        "${COMMON_ARGS[@]}" \
        --kl-divergence-base "${BASE}"
fi

run_compare f16   f16
run_compare q8_0  q8_0
run_compare q4_0  q4_0
run_compare nvfp4 nvfp4

run_compare nvfp4 f16
run_compare f16   nvfp4
run_compare q4_0  f16
run_compare f16   q4_0
