#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CHECKPOINT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66667 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU