from easydict import EasyDict

model_cfg = EasyDict()
model_cfg.name = 'TinyLLM'

model_cfg.dim = 512
model_cfg.d_ff = 2048
model_cfg.max_sequence_length = 1024
model_cfg.n_heads = 8
model_cfg.n_kv_heads = 4    # GQMA
model_cfg.layers_num = 6
model_cfg.eps = 1e-6
model_cfg.dropout_rate = 0.1

