#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0

PYTHON=${PYTHON:-"python3"}

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py --config-file $CONFIG ${@:3}
