#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PRETRAIN=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66667 \
    $(dirname "$0")/train.py $CONFIG --cfg-options model.pretrained=$PRETRAIN \
    --work-dir ./det-output --launcher pytorch ${@:4}
