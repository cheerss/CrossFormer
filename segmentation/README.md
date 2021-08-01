# CrossFormer Segmentation
Our semantic segmentation code is developed on top of [MMSegmentation v0.12.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0).

For more details please refer to our paper [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention]().




## Prerequisites

1. Libraries (Python3.6-based)
```bash
pip3 install mmcv-full==1.2.7 mmsegmentation==0.12.0
```

2. Prepare ADE20K dataset according to guidelines in [MMSegmentation v0.12.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0)

3. Prepare pretrained CrossFormer models
```python
import torch
ckpt = torch.load("pretrain-xxx.pth") ## load classification checkpoint
torch.save(ckpt["model"], "segm-pretrain-xxx.pth") ## only model weights are needed
```



## Getting Started

1. Modify `data_root` in `configs/_base_/datasets/ade20k.py`  and `configs/_base_/datasets/ade20k_swin.py` to your path to the ADE20K dataset.

2. Training
```bash
## Use config in Results table listed below as <CONFIG_FILE>
./dist_train.sh <CONFIG_FILE> <GPUS> <PRETRAIN_MODEL>

## e.g. train fpn_crossformer_b model with 8 GPUs
./dist_train.sh configs/fpn_crossformer_b_ade20k_40k.py 8 path/to/segm-pretrain-xxx.pth
```

3. Inference
```bash
./dist_test.sh <CONFIG_FILE> <GPUS> <DET_CHECKPOINT_FILE>

## e.g. evaluate semantic segmentation model by mIoU
./dist_test.sh configs/fpn_crossformer_b_ade20k_40k.py 8 path/to/ckpt
```
**Notes:** We use single-scale testing by default, you can enable multi-scale testing or flip testing manually by following the instructions in `configs/_base_/datasets/ade20k[_swin].py`.




## Results

### Semantic FPN

| Backbone      | Iterations | Params | FLOPs | IOU | config| Pretrained Model|
| ------------- | :-----: | ------:| -----:| ------:| -----:| ---------------:|
| PVT-M         | 80K    | 48.0M | 219.0G | 41.6  | - | - |
| PVT-L         | 80K    | 65.1M | 283.0G | 42.1  | - | - |
| Swin-S        | 80K    | 53.2M | 274.0G | 45.2  | - | - |
| CrossFormer-S | 80K    | 34.3M | 209.8G | 46.4  | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/fpn_crossformer_s_ade20k_40k.py)   | *TBD* |
| CrossFormer-B | 80K    | 55.6M | 320.1G | 48.0  | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/fpn_crossformer_b_ade20k_40k.py)   | *TBD* |
| CrossFormer-L | 80K    | 95.4M | 482.7G | **49.1** | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/fpn_crossformer_l_ade20k_40k.py)   | *TBD* |

### UPerNet

| Backbone      | Iterations | Params | FLOPs | IOU    | MS IOU | config| Pretrained Model|
| ------------- | :--------: | ------:| -----:| ------:| ------:| -----:| ---------------:|
| ResNet-101    | 160K   | 86.0M | 1029.0G | 44.9  | - | - | - |
| Swin-T        | 160K   | 60.0M | 945.0G  | 44.5  | 45.8 | - | - |
| Swin-S        | 160K   | 81.0M | 1038.0G | 47.6  | 49.5 | - | - |
| Swin-B        | 160K   | 121.0M| 1088.0G | 48.1  | 49.7 | - | - |
| CrossFormer-S | 160K   | 62.3M | 979.5G  | 47.6  | 48.4 | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/upernet_crossformer_s_ade20k.py)   | *TBD* |
| CrossFormer-B | 160K   | 83.6M | 1089.7G | 49.7  | 50.6 | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/upernet_crossformer_b_ade20k.py)   | *TBD* |
| CrossFormer-L | 160K   | 125.5M| 1257.8G | **50.4** | **51.4** | [config](https://github.com/cheerss/CrossFormer/blob/main/segmentation/configs/upernet_crossformer_l_ade20k.py)   | *TBD* |

**Notes:**
- *MS IOU* means *IOU* with multi-scale testing.
- Models are trained on ADE20K. Backbones are initialized with weights pre-trained on ImageNet-1K.
- For Semantic FPN, models are trained for 80K iterations with batch size 16. For UperNet, models are trained for 160K iterations.
- More detailed training settings can be found in corresponding configs.




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