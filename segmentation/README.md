# CrossFormer++ Segmentation

Our semantic segmentation code is developed on top of [MMSegmentation v0.29.1](https://github.com/open-mmlab/mmsegmentation/tree/v0.29.1).

For more details please refer to our paper [CrossFormer: A Versatile Vision Transformer Based on Cross-scale Attention](https://arxiv.org/pdf/2108.00154.pdf).




## Prerequisites

1. Libraries (Python3.6-based)
```bash
pip install mmcv-full==1.6.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install yapf==0.40.1
pip install mmsegmentation==0.29.1
```

2. Prepare ADE20K dataset according to guidelines in [MMSegmentation v0.12.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.12.0)

3. Prepare pretrained CrossFormer models
```python
import torch
ckpt = torch.load("crossformer-s.pth") ## load classification checkpoint
torch.save(ckpt["model"], "backbone-corssformer-s.pth") ## only model weights are needed
```



## Getting Started

1. Modify `data_root` in `configs/_base_/datasets/ade20k.py`  and `configs/_base_/datasets/ade20k_swin.py` to your path to the ADE20K dataset.

2. Training
```bash
## Use config in Results table listed below as <CONFIG_FILE>
./dist_train.sh <CONFIG_FILE> <GPUS> <PRETRAIN_MODEL>

## e.g. train fpn_crossformer_b model with 8 GPUs
./dist_train.sh configs/crossformer/fpn_crossformer_b_ade20k_40k.py 8 path/to/backbone-corssformer-s.pth
```

3. Inference
```bash
./dist_test.sh <CONFIG_FILE> <GPUS> <DET_CHECKPOINT_FILE>

## e.g. evaluate semantic segmentation model by mIoU
./dist_test.sh configs/crossformer/fpn_crossformer_b_ade20k_40k.py 8 path/to/ckpt
```
**Notes:** We use single-scale testing by default, you can enable multi-scale testing or flip testing manually by following the instructions in `configs/_base_/datasets/ade20k[_swin].py`.




## Results

### Semantic FPN

| Backbone      | Iterations | Params | FLOPs | IOU | config| Models|
| ------------- | :-----: | ------:| -----:| ------:| :-----| :---------------|
| PVT-M         | 80K    | 48.0M | 219.0G | 41.6  | - | - |
| **CrossFormer-S** | **80K**    | **34.3M** | **209.8G** | **46.4**  | [config](./configs/fpn_crossformer_s_ade20k_40k.py)   | [Google Drive](https://drive.google.com/file/d/1I-zpGG5rvkTtrTUnOF8Fx11yeb6pXGYi/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/14K3gJS3UcnEZNwhTWHdsFg), key: sn5h |
| PVT-L         | 80K    | 65.1M | 283.0G | 42.1  | - | - |
| Swin-S        | 80K    | 53.2M | 274.0G | 45.2  | - | - |
| **CrossFormer-B** | **80K**    | **55.6M** | **320.1G** | **48.0**  | [config](./configs/fpn_crossformer_b_ade20k_40k.py)   | [Google Drive](https://drive.google.com/file/d/1EjAnRc8Sau0un1ymqDVhFebHPeBIjukK/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1eCYENrLZsxjAQKW3eAQeUA), key: joi5 |
| **CrossFormer-L** | **80K**    | **95.4M** | **482.7G** | **49.1** | [config](./configs/fpn_crossformer_l_ade20k_40k.py)   | [Google Drive](https://drive.google.com/file/d/12WS9lX9yR5skxdt3N2HE3b2EaDSMwUMY/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/150b8-v1StaMHACIaM0hZVA), key: 6v5d |

### UPerNet

| Backbone      | Iterations | Params | FLOPs | IOU    | MS IOU | config| Models|
| ------------- | :--------: | ------:| -----:| ------:| ------:| :-----| :---------------|
| ResNet-101    | 160K   | 86.0M | 1029.0G | 44.9  | - | - | - |
| Swin-T        | 160K   | 60.0M | 945.0G  | 44.5  | 45.8 | - | - |
| **CrossFormer-S** | **160K**   | **62.3M** | **979.5G**  | **47.6**  | **48.4** | [config](./configs/upernet_crossformer_s_ade20k.py)   | [Google Drive](https://drive.google.com/file/d/1VKu4D6oxYdO1VVILLMhOQy8oCxKGH5gx/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1MLcAOiJ22AFKUa6_t1psOQ), key: wesb |
| Swin-S        | 160K   | 81.0M | 1038.0G | 47.6  | 49.5 | - | - |
| **CrossFormer-B** | **160K**   | **83.6M** | **1089.7G** | **49.7**  | **50.6** | [config](./configs/upernet_crossformer_b_ade20k.py)   | [Google Drive](https://drive.google.com/file/d/1B8VTNeidrzlfsOkQUKgmX4m_UfCIm58i/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1311pBQluwJGiVdWY1WE16Q), key: j061 |
| Swin-B        | 160K   | 121.0M| 1088.0G | 48.1  | 49.7 | - | - |
| **CrossFormer-L** | **160K**   | **125.5M**| **1257.8G** | **50.4** | **51.4** | [config](./configs/upernet_crossformer_l_ade20k.py)   | [Google Drive](https://drive.google.com/file/d/1I9ph5MeCwlTF2PNCkIYMFXYsdywp9nU1/view?usp=sharing)/[BaiduCloud](https://pan.baidu.com/s/1Yu8QB42hcbKNGQ46Wx_NaQ), key: 17ks |

**Notes:**
- *MS IOU* means *IOU* with multi-scale testing.
- Models are trained on ADE20K. Backbones are initialized with weights pre-trained on ImageNet-1K.
- For Semantic FPN, models are trained for 80K iterations with batch size 16. For UperNet, models are trained for 160K iterations.
- More detailed training settings can be found in corresponding configs.
- More results can be seen in our paper.




## FLOPs and Params Calculation
use `get_flops.py` to calculate FLOPs and #parameters of the specified model.
```bash
python get_flops.py <CONFIG_FILE> --shape <height> <width>

## e.g. get FLOPs and #params of fpn_crossformer_b with input image size [1024, 1024]
python get_flops.py configs/crossformer/fpn_crossformer_b_ade20k_40k.py --shape 1024 1024
```

**Notes:** Default input image size is [1024, 1024]. For calculation with different input image size, you need to change `<height> <width>` in the above command and change `img_size` in `crossformer_factory.py` accordingly at the same time.




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
