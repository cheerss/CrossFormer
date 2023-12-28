_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/default_runtime.py'
]
model = dict(
    pretrained=None,
    backbone=dict(
        type='CrossFormer_S2',
        group_size=[7, 7, 7, 7],
        crs_interval=[8, 4, 2, 1],
        adaptive_interval=False,
        init_cfg=dict(type='Pretrained', checkpoint='./backbone-crossformer-s.pth')),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
device = 'cuda'