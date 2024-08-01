from easydict import EasyDict

model_cfg = EasyDict()

model_cfg.dim = 512
model_cfg.d_ff = 2048
model_cfg.max_seq_len = 2048   # for RoPE

# en model
# model_cfg.name = 'TinyLLM'
# model_cfg.n_heads = 8
# model_cfg.n_kv_heads = 4       # GMQA
# model_cfg.n_layers = 12

# ru model
model_cfg.n_heads = 16
model_cfg.n_layers = 7
model_cfg.name = 'TinyLLM2'

model_cfg.eps = 1e-6
model_cfg.dropout_rate = 0.1
model_cfg.rope_theta = 100_000.0
model_cfg.multiple_of = 256
model_cfg.ffn_dim_multiplier = None
model_cfg.norm_eps = 1e-5
