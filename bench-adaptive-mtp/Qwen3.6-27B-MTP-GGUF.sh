#!/bin/bash

# Global
export LLAMA_ARG_N_PREDICT=-1
export LLAMA_ARG_UBATCH=512
export LLAMA_ARG_BATCH=2048
export LLAMA_ARG_CACHE_TYPE_K=q8_0
export LLAMA_ARG_CACHE_TYPE_V=q4_0
export LLAMA_ARG_N_PARALLEL=1

# Local
export LLAMA_ARG_N_GPU_LAYERS=all
export LLAMA_ARG_N_GPU_LAYERS_DRAFT=all
export LLAMA_ARG_SWA_FULL=1
export LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_K=q8_0
export LLAMA_ARG_SPEC_DRAFT_CACHE_TYPE_V=q8_0

# Multimodal
export LLAMA_ARG_NO_MMPROJ=1
export LLAMA_ARG_MMPROJ_OFFLOAD=0


# Qwn 3.6 specifcic
export LLAMA_ARG_CHAT_TEMPLATE_KWARGS='{"preserve_thinking":true}'


# speculative decoding
 ## old: 2=116.0 3=124.2 4=140.6 5=137.6 (short prompt)
 ## long-response recitation sweep (Adaptive_MTP.md): head degrades gracefully with depth,
 ## n=16/p=0.8 ~225 t/s (+60% over n=4); plateaus past 16, no collapse through 24
export LLAMA_ARG_SPEC_DRAFT_N_MAX=16
export LLAMA_ARG_SPEC_DRAFT_P_MIN=0.8
export LLAMA_ARG_SPEC_TYPE=draft-mtp

command -v llama-server

llama-server -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q6_K -fa on -ub 1024 \
--temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 \
--alias LocalLLM "$@"

#LLAMA_ARG_BATCH=8192 LLAMA_ARG_UBATCH=1024  n_ctx = 168960
#LLAMA_ARG_BATCH=8192 LLAMA_ARG_UBATCH=1024 LLAMA_ARG_NO_MMPROJ=1 n_ctx = 205568
#LLAMA_ARG_BATCH=8192 LLAMA_ARG_UBATCH=2048 n_ctx = 100864
