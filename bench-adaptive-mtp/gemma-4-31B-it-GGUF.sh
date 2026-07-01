#!/bin/bash

# Global
export LLAMA_ARG_N_PREDICT=-1
export LLAMA_ARG_UBATCH=512
export LLAMA_ARG_BATCH=2048
export LLAMA_ARG_CACHE_TYPE_K=q8_0
export LLAMA_ARG_CACHE_TYPE_V=q4_0
export LLAMA_ARG_N_PARALLEL=1
export LLAMA_ARG_N_PARALLEL=2

# Local
export LLAMA_ARG_N_GPU_LAYERS=all
export LLAMA_ARG_N_GPU_LAYERS_DRAFT=all
export LLAMA_ARG_SWA_FULL=0 # OOM reason
export LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K=q8_0
export LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V=q8_0

# Tweaks
export LLAMA_ARG_NO_MMPROJ=1
export LLAMA_ARG_MMPROJ_OFFLOAD=0

# speculative decoding
 ## n_max x p_min sweep (long-response recitation): see Adaptive_MTP.md
 ## n=16/p=0.8 is the peak (~2.7x the n=1 floor, ~5x no-spec); degrades past ~16.
 ## p_min gates per-token, so a high fixed cap isn't penalized on reasoning content.
export LLAMA_ARG_SPEC_DRAFT_N_MAX=16
export LLAMA_ARG_SPEC_DRAFT_P_MIN=0.8
export LLAMA_ARG_SPEC_TYPE=draft-mtp


command -v llama-server

llama-server -hf unsloth/gemma-4-31B-it-GGUF:UD-Q4_K_XL \
--temp 0.6 --top-p 0.95 --top-k 20 \
--alias LocalLLM "$@"


