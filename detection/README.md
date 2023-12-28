# CrossFormer++ Detection

Our detection code is developed on top of [MMDetection v2.26.0](https://github.com/open-mmlab/mmdetection/tree/v2.26.0).

For more details please refer to our paper [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](https://arxiv.org/pdf/2108.00154.pdf).




## Prerequisites

1. Libraries (Python3.6-based)
```bash
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install yapf==0.40.1
pip install mmdet==2.26.0
```

2. Prepare COCO 2017 dataset according to guidelines in [MMDetection v2.26.0](https://github.com/open-mmlab/mmdetection/tree/v2.26.0)

3. Prepare CrossFormer models pre-trained on ImageNet-1K
```python
import torch
ckpt = torch.load("crossformer-s.pth") ## load classification checkpoint
torch.save(ckpt["model"], "backbone-crossformer-s.pth") ## only model weights are needed
```




## Getting Started

1. Modify `data_root` in `configs/_base_/datasets/coco_detection.py`  and `detection/configs/_base_/datasets/coco_instance.py` to your path to the COCO dataset.

2. Training
```bash
## Use config in Results table listed below as <CONFIG_FILE>
./dist_train.sh <CONFIG_FILE> <GPUS> <PRETRAIN_MODEL>

## e.g. train model with 8 GPUs
./dist_train.sh configs/crossformer/retinanet_crossformer_s_fpn_1x_coco.py 8 path/to/backbone-crossformer-s.pth
```

3. Inference
```bash
./dist_test.sh <CONFIG_FILE> <GPUS> <DET_CHECKPOINT_FILE> --eval bbox [segm]

## e.g. evaluate detection model
./dist_test.sh configs/crossformer/retinanet_crossformer_s_fpn_1x_coco.py 8 path/to/ckpt --eval bbox

## e.g. evaluate instance segmentation model
./dist_test.sh configs/crossformer/mask_rcnn_crossformer_s_fpn_1x_coco.py 8 path/to/ckpt --eval bbox segm
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
| ------------- | :-----: |-------:| ------:| ------:| -------:| :-----| :---------------|
| ResNet-101    | 1x      | 63.2M  | 336.0G | 40.4   | 36.4 | - | - |
| PVT-M         | 1x      | 63.9M  | -      | 42.0   | 39.0 | - | - |
| Swin-T        | 1x      | 47.8M  | 264.0G | 42.2   | 39.1 | - | - |
| **CrossFormer-S** | **1x**      | **50.2M**  | **301.0G** | **45.4**   | **41.4** | [config](./configs/mask_rcnn_crossformer_s_fpn_1x_coco.py) | [Google Drive](https://drive.google.com/file/d/1qgO3-wnsJQw9Nj304AOj-TjVyadPPtvf/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1ytk95p-rjqI3UNEHaYqKKA), key: 4omu |
| PVT-L         | 1x      | 81.0M  | 364.0G | 42.9   | 39.5 | - | - |
| Swin-B        | 1x      | 107.2M | 496.0G | 45.5   | 41.3 | - | - |
| **CrossFormer-B** | **1x**      | **71.5M**  | **407.9G** | **47.2** | **42.7** | [config](./configs/mask_rcnn_crossformer_b_fpn_1x_coco.py) | [Google Drive](https://drive.google.com/file/d/1urkCa8Inss9CL1G1OgM4tWR5PSf6Ofn4/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1N_EGRLBuIXgQh6J0Oyh00g), key: kjf4 |
| **CrossFormer-S** | **3x**  | **50.2M**  | **291.1G** | **48.7**   | **43.9** | - | - |
| **CrossFormer-B** | **3x**  | **71.5M**  | **398.1G** | **49.8**   | **44.5** | - | - |

### Cascade Mask R-CNN
| Backbone      | Lr schd | Params | FLOPs  | box AP | mask AP | config| Models |
| ------------- | :-----: |-------:| ------:| ------:| -------:| :-----| :---------------|
| **CrossFormer-S** | **3x**  | **88.0M**  | **769.7G** | **52.2**   | **45.2** | - | - |


**Notes:**
- Models are trained on the COCO train2017 (~118k images) and evaluated on the val2017(5k images). Backbones are initialized with weights pre-trained on ImageNet-1K.
- Models are trained with batch size 16 on 8 V100 GPUs.
- We adopt 1x training schedule, *i.e.*, the models are trained for 12 epochs.
- The training image is resized to the shorter side of 800 pixels, while the longer side does not exceed 1333 pixels.
- More detailed training settings can be found in corresponding configs.
- More results can be seen in our paper.




## FLOPs and Params Calculation
use `get_flops.py` to calculate FLOPs and #parameters of the specified model.
```bash
python get_flops.py <CONFIG_FILE> --shape <height> <width>

## e.g. get FLOPs and #params of retinanet_crossformer_s with input image size [1280, 800]
python get_flops.py configs/crossformer/retinanet_crossformer_s_fpn_1x_coco.py --shape 1280 800
```

**Notes:** Default input image size is [1280, 800]. For calculation with different input image size, you need to change `<height> <width>` in the above command and change `img_size` in `crossformer_factory.py` accordingly at the same time.




## Citing Us

```
@inproceedings{wang2021crossformer,
  title = {CrossFormer: A Versatile Vision Transformer Hinging on Cross-scale Attention},
  author = {Wang, Wenxiao and Yao, Lu and Chen, Long and Lin, Binbin and Cai, Deng and He, Xiaofei and Liu, Wei},
  booktitle = {International Conference on Learning Representations, {ICLR}},
  url = {https://openreview.net/forum?id=_PHymLIxuI},
  year = {2022}
}

@article{wang2023crossformer++,
  title={Crossformer++: A versatile vision transformer hinging on cross-scale attention},
  author={Wang, Wenxiao and Chen, Wei and Qiu, Qibo and Chen, Long and Wu, Boxi and Lin, Binbin and He, Xiaofei and Liu, Wei},
  journal={arXiv preprint arXiv:2303.06908},
  year={2023}
}
```




