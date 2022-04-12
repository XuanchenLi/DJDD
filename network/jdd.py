import torch
import torch.nn as nn
import collections
import numpy as np

class BayerJDDNetwork(nn.Module):
    def __init__(self, width=64, depth=16, pre_trained=False, padding=True):
        super(BayerJDDNetwork, self).__init__()

        self.width = width
        self.depth = depth

        # 下采样
        self.down_sample = nn.Conv2d(3, 4, (2, 2), stride=(2, 2))
        # D-1次卷积
        self.layers = collections.OrderedDict()
        if padding:
            padding = 1
        else:
            padding = 0
        for i in range(depth - 1):
            in_size = width
            out_size = width
            if i == 0:
                in_size = 5
            self.layers["Conv_{}".format(i + 1)] = \
                nn.Conv2d(in_size, out_size, kernel_size=(3, 3), padding=padding)
            self.layers["ReLU_{}".format(i + 1)] = nn.ReLU(inplace=True)

        self.main_layers = nn.Sequential(self.layers)
        # residual层
        self.residual = nn.Conv2d(width, 12, (1, 1))
        # 上采样
        self.up_sample = nn.ConvTranspose2d(12, 3, (2, 2), stride=(2, 2), groups=3)

        self.final_process = nn.Sequential(
            nn.Conv2d(6, width, (3, 3), padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, (1, 1))
        )

    def forward(self, inputs, noise_level):
        F0 = self.down_sample(inputs)
        p_size = F0.size()[2:]
        noise = torch.stack([torch.full(p_size, noi) for noi in noise_level]).unsqueeze(1)
        F0 = torch.cat((noise, F0), dim=1)
        features = self.main_layers(F0)
        residual = self.residual(features)
        FD1 = self.up_sample(residual)
        FD1 = torch.cat((inputs, FD1), dim=1)
        outputs = self.final_process(FD1)
        return outputs






