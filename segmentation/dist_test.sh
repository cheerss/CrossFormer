#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=66667 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU