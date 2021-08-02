# CrossFormer Detection

Our detection code is developed on top of [MMDetection v2.8.0](https://github.com/open-mmlab/mmdetection/tree/v2.8.0).

For more details please refer to our paper [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention]().




## Prerequisites

1. Libraries (Python3.6-based)
```bash
pip3 install mmcv-full==1.2.7 mmdet==2.8.0
```

2. Prepare COCO 2017 dataset according to guidelines in [MMDetection v2.8.0](https://github.com/open-mmlab/mmdetection/tree/v2.8.0)

3. Prepare CrossFormer models pre-trained on ImageNet-1K
```python
import torch
ckpt = torch.load("crossformer-s.pth") ## load classification checkpoint
torch.save(ckpt["model"], "pretrain-corssformer-s.pth") ## only model weights are needed
```




## Getting Started

1. Modify `data_root` in `configs/_base_/datasets/coco_detection.py`  and `detection/configs/_base_/datasets/coco_instance.py` to your path to the COCO dataset.

2. Training
```bash
## Use config in Results table listed below as <CONFIG_FILE>
./dist_train.sh <CONFIG_FILE> <GPUS> <PRETRAIN_MODEL>

## e.g. train model with 8 GPUs
./dist_train.sh configs/retinanet_crossformer_s_fpn_1x_coco.py 8 path/to/pretrain-corssformer-s.pth
```

3. Inference
```bash
./dist_test.sh <CONFIG_FILE> <GPUS> <DET_CHECKPOINT_FILE> --eval bbox [segm]

## e.g. evaluate detection model
./dist_test.sh configs/retinanet_crossformer_s_fpn_1x_coco.py 8 path/to/ckpt --eval bbox

## e.g. evaluate instance segmentation model
./dist_test.sh configs/mask_rcnn_crossformer_s_fpn_1x_coco.py 8 path/to/ckpt --eval bbox segm
```




## Results

### RetinaNet

| Backbone      | Lr schd | Params | FLOPs | box AP | config| Models |
| ------------- | :-----: | ------:| -----:| ------:| :-----| :---------------|
| ResNet-101    | 1x      | 56.7M  | 315.0G   | 38.5     | - | - |
| PVT-M         | 1x      | 53.9M  | -       | 41.9     | - | - |
| Swin-T        | 1x      | 38.5M  | 245.0G   | 41.5     | - | - |
| **CrossFormer-S** | 1x      | **40.8M**  | **282.0G**   | **44.4**     | [config](./configs/retinanet_crossformer_s_fpn_1x_coco.py)   | [Google Drive](https://drive.google.com/file/d/1OEEottS4fYGVZcPZG6WuBlKMK0tZLPZW/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1Ckrk3Z1uA65ve43_hL3ZFg), key: ift3 |
| PVT-L         | 1x      | 71.1M  | 345.0G   | 42.6     | - | - |
| Swin-B        | 1x      | 98.4M  | 477.0G   | 44.7     | - | - |
| **CrossFormer-B** | **1x**      | **62.1M**  | **389.0G**   | **46.2** | [config](./configs/retinanet_crossformer_b_fpn_1x_coco.py)   | [Google Drive](https://drive.google.com/file/d/1TOuLLf_S4Ixo6COvaHQAocACR4NKKnGz/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1ESE7i1JpVjxZmB5ZTgXA5w), key: hsd5 |


### Mask R-CNN

| Backbone      | Lr schd | Params | FLOPs  | box AP | mask AP | config| Models |
| ------------- | :-----: |-------:| ------:| ------:| -------:| -----:| ---------------:|
| ResNet-101    | 1x      | 63.2M  | 336.0G | 40.4   | 36.4 | - | - |
| PVT-M         | 1x      | 63.9M  | -      | 42.0   | 39.0 | - | - |
| Swin-T        | 1x      | 47.8M  | 264.0G | 42.2   | 39.1 | - | - |
| **CrossFormer-S** | **1x**      | **50.2M**  | **301.0G** | **45.4**   | **41.4** | [config](./configs/mask_rcnn_crossformer_s_fpn_1x_coco.py) | *TBD* |
| PVT-L         | 1x      | 81.0M  | 364.0G | 42.9   | 39.5 | - | - |
| Swin-B        | 1x      | 107.2M | 496.0G | 45.5   | 41.3 | - | - |
| **CrossFormer-B** | **1x**      | **71.5M**  | **407.9G** | **47.2** | **42.7** | [config](./configs/mask_rcnn_crossformer_b_fpn_1x_coco.py) | *TBD* |


**Notes:**
- Models are trained on the COCO train2017 (~118k images) and evaluated on the val2017(5k images). Backbones are initialized with weights pre-trained on ImageNet-1K.
- Models are trained with batch size 16 on 8 V100 GPUs.
- We adopt 1x training schedule, *i.e.*, the models are trained for 12 epochs.
- The training image is resized to the shorter side of 800 pixels, while the longer side does not exceed 1333 pixels.
- More detailed training settings can be found in corresponding configs.
- More results can be seen in our paper.




## Citing Us

```
@article{crossformer2021,
  title     = {CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention},
  author    = {Wenxiao Wang and Lu Yao and Long Chen and Deng Cai and Xiaofei He and Wei Liu},
  journal   = {CoRR},
  volume    = {abs/21xx.xxxxx},
  year      = {2021},
}
```




