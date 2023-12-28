#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
CHECKPOINT=$3

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=66667 \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU \
    --options model.backbone.group_size=8,8,14,28 model.backbone.crs_interval=16,8,4,1 model.backbone.adaptive_interval=False