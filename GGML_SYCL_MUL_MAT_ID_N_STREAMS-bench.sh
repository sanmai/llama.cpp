#!/usr/bin/env bash
set -euo pipefail

for n_streams in 1 2 4 8; do
    echo GGML_SYCL_MUL_MAT_ID_N_STREAMS="${n_streams}"
    GGML_SYCL_MUL_MAT_ID_N_STREAMS="${n_streams}" ./build-local/bin/llama-bench -m OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf -ngl 999 -p 2048 -n 128 -b 4096 -ub 2048 -r 5
done
