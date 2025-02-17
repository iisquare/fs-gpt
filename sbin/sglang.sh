#!/bin/bash

# https://docs.sglang.ai/backend/server_arguments.html

export CUDA_VISIBLE_DEVICES=1

nohup python -m sglang.launch_server --model-path ./models/Qwen2.5-32B-Instruct-GPTQ-Int4 \
--trust-remote-code \
--tp 1 \
--mem_fraction_static 0.9 \
--host 0.0.0.0 \
--port 6011 \
> ./logs/sglang.log 2>&1 &
