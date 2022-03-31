import numpy as np
import torch


def bayer(img, return_mask=False):
    """
    G B
    R G
    """
    mask = torch.ones_like(img)

    # red
    mask[..., 0, ::2, :] = 0
    mask[..., 0, 1::2, 1::2] = 0

    # green
    mask[..., 1, 0::2, 1::2] = 0
    mask[..., 1, 1::2, 0::2] = 0

    # blue
    mask[..., 2, 1::2, :] = 0
    mask[..., 2, 0::2, 0::2] = 0

    if return_mask:
        return mask
    return img * mask