#!/bin/bash

CONFIG_FILE=/etc/vllm/config.yaml

function parse() {
  cat $1 | while read LINE
  do
    if [ "$(echo $LINE | grep -E ' ')" != "" ];then
      echo "$LINE" | awk -F ": " '{
        key = $1;
        value = $2;
        gsub(/^\s+|\s+$/, "", key);
        gsub(/^\s+|\s+$/, "", value);
        if (value == "True") {
          printf " --%s", key
        } else if (value != "False") {
          printf " --%s %s", key, value
        }
      }'
    fi
  done
}

cmd=$(cat <<- EOF
vllm serve \
$(parse $CONFIG_FILE)
EOF
)

eval $cmd
