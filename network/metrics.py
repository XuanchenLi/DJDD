import numpy as np
import torch as th


class PSNR(th.nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = th.nn.MSELoss()

    def forward(self, out, ref):
        mse = self.mse(out, ref)
        return -10 * th.log10(mse + 1e-12)


