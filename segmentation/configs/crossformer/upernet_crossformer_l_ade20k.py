_base_ = [
    '../_base_/models/upernet_crossformer.py', 
    '../_base_/datasets/ade20k_swin.py',
    '../_base_/default_runtime.py', 
    '../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained=None,
    backbone=dict(type='CrossFormer_L', group_size=[7, 7, 7, 7], crs_interval=[8, 4, 2, 1],
        init_cfg=dict(type='Pretrained', checkpoint='./backbone-crossformer-s.pth')),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024]
    ),
    auxiliary_head=dict(
        in_channels=512 # 3rd in_channels of decode_head
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

