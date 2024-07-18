import math

import torch
from torch import nn


def custom_lr_schedule(cur_step, emb_size=512, warmup_steps=500):
    """Constructs custom learning rate schedule.

    Learning rate changes during the training process should follow the next formula:
        learning_rate = d_model^(-0.5) * min((cur_step + 1)^(-0.5), (cur_step + 1) * warmup_steps^(-1.5)
    """
    learning_rate = emb_size ** (-0.5) * min((cur_step + 1) ** (-0.5), (cur_step + 1) * warmup_steps ** (-1.5))
    return learning_rate


def interpolate_pos_embeddings(pos_embeddings, d_model, patch_size, num_patches, w, h):
    """Positional embeddings interpolation.

    This method interpolates positional embeddings for the cases when input images have a different size
            than the one used during training.
    """
    pretrain_num_patches = pos_embeddings.shape[1] - 1
    if num_patches == pretrain_num_patches:
        return pos_embeddings

    class_pos_embed = pos_embeddings[:, 0]
    patch_pos_embeds = pos_embeddings[:, 1:]

    N_w, N_h = w // patch_size, h // patch_size
    N_sqrt = math.sqrt(pretrain_num_patches)

    new_patch_pos_embeds = nn.functional.interpolate(
        patch_pos_embeds.reshape(1, int(N_sqrt), int(N_sqrt), d_model).permute(0, 3, 1, 2),
        scale_factor=(N_w / N_sqrt, N_h / N_sqrt),
        mode='bicubic',
    )
    new_patch_pos_embeds = new_patch_pos_embeds.permute(0, 2, 3, 1).view(1, -1, d_model)

    return torch.cat((class_pos_embed.unsqueeze(0), new_patch_pos_embeds), dim=1)
