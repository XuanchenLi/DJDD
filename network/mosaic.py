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


def bayer_down_sample(src):
    new_size = list(src.shape)
    new_size[1] = 4
    new_size[2] = int(new_size[2]/2)
    new_size[3] = int(new_size[3]/2)
    res = torch.zeros(tuple(new_size))
    res[:, 0, :, :] = src[:, 1, ::2, ::2]  # G
    res[:, 1, :, :] = src[:, 0, 1::2, ::2]  # R
    res[:, 2, :, :] = src[:, 2, ::2, 1::2]  # B
    res[:, 3, :, :] = src[:, 1, 1::2, 1::2]  # G
    return res


def bayer_up_sample(src):
    new_size = list(src.shape)
    new_size[1] = 3
    new_size[2] *= 2
    new_size[3] *= 2
    # print(src)
    res = torch.zeros(tuple(new_size))
    for c in range(3):
        res[:, c, ::2, ::2] = src[:, 4*c, :, :]
        res[:, c, ::2, 1::2] = src[:, 4*c+1, :, :]
        res[:, c, 1::2, ::2] = src[:, 4*c+2, :, :]
        res[:, c, 1::2, 1::2] = src[:, 4*c+3, :, :]
    return res
