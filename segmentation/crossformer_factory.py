import sys
import os
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# from yacs.config import CfgNode as CN
from mmseg.models.builder import BACKBONES

# from config import _C, _update_config_from_file
from models.crossformer_backbone_seg import CrossFormer


@BACKBONES.register_module()
class CrossFormer_S(CrossFormer):
    def __init__(self, **kwargs):
        super(CrossFormer_S, self).__init__(
            img_size=[1024,1024], # This is only used to compute the FLOPs under the give image size
            patch_size=[4, 8, 16, 32],
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            group_size=kwargs["group_size"],
            crs_interval=kwargs["crs_interval"],
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2,4], [2,4], [2,4]],
            use_acl=False,
            init_cfg=kwargs["init_cfg"]
        )

@BACKBONES.register_module()
class CrossFormer_S2(CrossFormer):
    def __init__(self, **kwargs):
        super(CrossFormer_S2, self).__init__(
            img_size=[1024,1024], # This is only used to compute the FLOPs under the give image size
            patch_size=[4, 8, 16, 32],
            in_chans=3,
            num_classes=1000,
            embed_dim=64,
            depths=[2, 2, 18, 2],
            num_heads=[2, 4, 8, 16],
            group_size=kwargs["group_size"],
            crs_interval=kwargs["crs_interval"],
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.2,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2,4], [2,4], [2,4]],
            adaptive_interval=kwargs["adaptive_interval"],
            use_acl=True,
            init_cfg=kwargs["init_cfg"]
        )

@BACKBONES.register_module()
class CrossFormer_B(CrossFormer):
    def __init__(self, **kwargs):
        super(CrossFormer_B, self).__init__(
            img_size=[1024,1024],
            patch_size=[4, 8, 16, 32],
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            group_size=kwargs["group_size"],
            crs_interval=kwargs["crs_interval"],
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2,4], [2,4], [2,4]],
            use_acl=False,
            init_cfg=kwargs["init_cfg"]
        )

@BACKBONES.register_module()
class CrossFormer_B2(CrossFormer):
    def __init__(self, **kwargs):
        super(CrossFormer_B, self).__init__(
            img_size=[1024,1024],
            patch_size=[4, 8, 16, 32],
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depths=[2, 2, 18, 2],
            num_heads=[3, 6, 12, 24],
            group_size=kwargs["group_size"],
            crs_interval=kwargs["crs_interval"],
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.3,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2,4], [2,4], [2,4]],
            use_acl=True,
            init_cfg=kwargs["init_cfg"]
        )

@BACKBONES.register_module()
class CrossFormer_L(CrossFormer):
    def __init__(self, **kwargs):
        super(CrossFormer_L, self).__init__(
            img_size=[1024,1024],
            patch_size=[4, 8, 16, 32],
            in_chans=3,
            num_classes=1000,
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            group_size=kwargs["group_size"],
            crs_interval=kwargs["crs_interval"],
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            drop_path_rate=0.5,
            patch_norm=True,
            use_checkpoint=False,
            merge_size=[[2,4], [2,4], [2,4]],
            use_acl=False,
            init_cfg=kwargs["init_cfg"]
        )
        
