#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0,1

CONFIG=$1
GPUS=$2
PRETRAIN=$3

python preprocess.py --ckpt_path $PRETRAIN

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=62345 \
    $(dirname "$0")/train.py $CONFIG \
    --work-dir ./seg-output --launcher pytorch ${@:4} --seed 0
