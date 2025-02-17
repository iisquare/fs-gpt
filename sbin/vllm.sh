#!/bin/bash

# https://docs.vllm.ai/en/latest/serving/engine_args.html

export CUDA_VISIBLE_DEVICES=1

nohup vllm serve ./models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--gpu-memory-utilization 0.9 \
--max-num-seqs 8 \
--enforce-eager \
--host 0.0.0.0 \
--port 6011 \
> ./logs/vllm.log 2>&1 &
