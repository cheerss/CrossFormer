_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        type='CrossFormer_S', group_size=[7, 7, 7, 7], crs_interval=[8, 4, 2, 1], adaptive_interval=False,
        init_cfg=dict(type='Pretrained', checkpoint='./backbone-crossformer-s.pth')),
    neck=dict(in_channels=[96,192,384,768],
    )
)


gpu_multiples = 1
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
device = 'cuda'