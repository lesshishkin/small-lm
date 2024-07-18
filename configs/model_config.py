from easydict import EasyDict

model_cfg = EasyDict()
model_cfg.name = 'PreTrainedVisionTransformer' # from ('Transformer', 'VisionTransformer', 'PreTrainedVisionTransformer')

model_cfg.d_model = 512
model_cfg.d_ff = 2048
model_cfg.max_sequence_length = 1024
model_cfg.heads_num = 8
model_cfg.layers_num = 6
model_cfg.eps = 1e-5
model_cfg.dropout_rate = 0.1
model_cfg.pre_normalization = True
model_cfg.activation = 'GELU'  # from (ReLU, GELU)
model_cfg.attention_bias = False

model_cfg.vit = EasyDict()
model_cfg.vit.name = model_cfg.name
model_cfg.vit.size = 'small'  # from ('tiny', 'small')
model_cfg.vit.feature_extraction = True
model_cfg.vit.pretrained_model_checkpoint_path = ''
model_cfg.vit.eps = 1e-5
model_cfg.vit.dropout_rate = 0
model_cfg.vit.pre_normalization = True
model_cfg.vit.activation = 'GELU'  # ReLU
model_cfg.vit.patch_size = 14
model_cfg.vit.image_size = 224 # from (128, 160, 192, 224)
model_cfg.vit.d_model = {'tiny': 192, 'small': 384}[model_cfg.vit.size]
model_cfg.vit.heads_num = {'tiny': 3, 'small': 6}[model_cfg.vit.size]
model_cfg.vit.layers_num = {'tiny': 12, 'small': 12}[model_cfg.vit.size]  # May vary for large and huge model sizes
model_cfg.vit.d_ff = model_cfg.vit.d_model * 4
model_cfg.vit.attention_bias = True
