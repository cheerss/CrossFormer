def build_model(config, args):
    model_type = config.MODEL.TYPE
    if model_type == 'cross-scale':
        if config.MODEL.CROS.GROUP_TYPE in ['constant', 'linear', 'linear_div', 'alter', '7_14']:
            from .crossformer import CrossFormer
        else:
            raise NotImplementedError(f"Unkown group type: {args.group_type}")

        model = CrossFormer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.CROS.PATCH_SIZE,
                                in_chans=config.MODEL.CROS.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.CROS.EMBED_DIM,
                                depths=config.MODEL.CROS.DEPTHS,
                                num_heads=config.MODEL.CROS.NUM_HEADS,
                                group_size=config.MODEL.CROS.GROUP_SIZE,
                                crs_interval=config.MODEL.CROS.INTERVAL,
                                mlp_ratio=config.MODEL.CROS.MLP_RATIO,
                                qkv_bias=config.MODEL.CROS.QKV_BIAS,
                                qk_scale=config.MODEL.CROS.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.CROS.APE,
                                patch_norm=config.MODEL.CROS.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                merge_size=config.MODEL.CROS.MERGE_SIZE,
                                group_type=config.MODEL.CROS.GROUP_TYPE,
                                use_cpe=config.MODEL.CROS.USE_CPE,
                                pad_type=config.MODEL.CROS.PAD_TYPE,
                                no_mask=config.MODEL.CROS.NO_MASK,
                                adaptive_interval=config.MODEL.CROS.ADAPT_INTER,
                                use_acl=config.MODEL.CROS.USE_ACL)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
