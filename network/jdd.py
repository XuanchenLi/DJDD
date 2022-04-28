import torch
import torch.nn as nn
import collections
from torchvision import transforms
import numpy as np


class BayerJDDNetwork(nn.Module):
    def __init__(self, width=64, depth=16, pre_trained=False, padding=True):
        super(BayerJDDNetwork, self).__init__()

        self.width = width
        self.depth = depth

        # 下采样
        self.down_sample = nn.Conv2d(3, 4, (2, 2), stride=(2, 2))
        torch.nn.init.kaiming_uniform_(
            self.down_sample.weight,
            a=0, mode='fan_in', nonlinearity='relu'
        )
        self.down_sample.bias.data.zero_()
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
            """
            if i == depth - 2:
                out_size = self.width * 2
            """
            self.layers["Conv_{}".format(i + 1)] = \
                nn.Conv2d(in_size, out_size, kernel_size=(3, 3), padding=padding)
            torch.nn.init.kaiming_uniform_(
                self.layers["Conv_{}".format(i + 1)].weight,
                a=0, mode='fan_in', nonlinearity='relu'
            )
            nn.init.constant_(self.layers["Conv_{}".format(i + 1)].bias, 0)
            self.layers["ReLU_{}".format(i + 1)] = nn.ReLU(inplace=True)

        self.main_layers = nn.Sequential(self.layers)
        # residual层
        self.residual = nn.Conv2d(width, 12, (1, 1))
        nn.init.constant_(self.residual.bias, 0)
        torch.nn.init.kaiming_uniform_(
            self.residual.weight,
            a=0, mode='fan_in', nonlinearity='relu'
        )
        # 上采样
        self.up_sample = nn.ConvTranspose2d(12, 3, (2, 2), stride=(2, 2), groups=3)
        nn.init.constant_(self.up_sample.bias, 0)
        torch.nn.init.kaiming_uniform_(
            self.up_sample.weight,
            a=0, mode='fan_in', nonlinearity='relu'
        )
        self.final_process = nn.Sequential(
            nn.Conv2d(6, width, (3, 3), padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, (1, 1))
        )
        for i in range(len(self.final_process)):
            if i % 2 == 0:
                nn.init.constant_(self.final_process[i].bias, 0)
                torch.nn.init.kaiming_uniform_(
                    self.final_process[i].weight,
                    a=0, mode='fan_in', nonlinearity='relu'
                )

    def forward(self, inputs, noise_level):
        F0 = self.down_sample(inputs)
        # F0 = inputs
        p_size = F0.size()[2:]
        noise = torch.stack([torch.full(p_size, noi) for noi in noise_level]).unsqueeze(1)
        # noise = torch.stack([torch.full(p_size, noi) for noi in noise_level]).unsqueeze(1).cuda()
        F0 = torch.cat((noise, F0), dim=1)
        # print(noise[0], F0[0][0])
        features = self.main_layers(F0)
        # filters, masks = features[:, 0:self.width], features[:, self.width:2 * self.width]
        # filtered = filters * masks
        residual = self.residual(features)
        # residual = self.residual(filtered)
        FD1 = self.up_sample(residual)
        """
        toPIL = transforms.ToPILImage()
        img = toPIL(FD1[0])
        img.show()
        """
        FD1 = torch.cat((inputs, FD1), dim=1)
        outputs = self.final_process(FD1)
        return torch.clamp(outputs, 0, 1)








