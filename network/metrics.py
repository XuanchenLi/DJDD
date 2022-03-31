import numpy as np
import torch


def psnr(tar, ref):
    target = np.array(tar, dtype=np.float64)
    refer = np.array(ref, dtype=np.float64)
    diff = refer - target
    diff = diff.flatten('C')
    rmse = np.sqrt(diff, np.mean(diff ** 2))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        return eps
    return 20 * np.log10(255.0/rmse)

