#!/bin/bash

# https://docs.vllm.ai/en/latest/serving/engine_args.html

PROJECT_ROOT=$(dirname $(dirname $(readlink -f $0)))
MODEL_PATH=${PROJECT_ROOT}/models/Qwen2.5-0.5B

export CUDA_VISIBLE_DEVICES=0

nohup vllm serve ${MODEL_PATH} \
--served-model-name $(basename ${MODEL_PATH}) \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--gpu-memory-utilization 0.9 \
--max-num-seqs 8 \
--enforce-eager \
--host 0.0.0.0 \
--port 6011 \
> ${PROJECT_ROOT}/logs/vllm.log 2>&1 &
