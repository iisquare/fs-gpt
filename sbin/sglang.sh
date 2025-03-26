#!/bin/bash

# https://docs.sglang.ai/backend/server_arguments.html

PROJECT_ROOT=$(dirname $(dirname $(readlink -f $0)))
MODEL_PATH=${PROJECT_ROOT}/models/Qwen2.5-0.5B

export CUDA_VISIBLE_DEVICES=0

cmd=$(cat <<- EOF
python -m sglang.launch_server --model-path ${MODEL_PATH} \
--trust-remote-code \
--tp 1 \
--host 0.0.0.0 \
--port 6011
EOF
)

if [ "-d" = "$1" ]; then
  echo "run in daemon model..."
  cmd="nohup ${cmd} > ${PROJECT_ROOT}/logs/sglang.log 2>&1 &"
fi

eval $cmd
