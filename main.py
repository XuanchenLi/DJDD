import numpy as np
import os
import torch as th
import torch.utils.data as Data
from torchvision import transforms
from network.data import DemosaicDataset
from network.jdd import BayerJDDNetwork
from network.trainer import Demosaicnet
from PIL import Image
from PIL import ImageFile


BATCH_SIZE = 8
EPOCHS = 3

if __name__ == '__main__':
    th.backends.cudnn.enabled = False
    th.cuda.empty_cache()
    """
    dd = ['dataset/train/moire/000', 'dataset/val/moire/000', 'dataset/val/hdrvdp/000']
    model = BayerJDDNetwork()
    model.cuda()
    # t_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(t_num)
    trainer = Demosaicnet(model)
    trainer.train(dd, BATCH_SIZE, EPOCHS)
    trainer.save_model()
    """
    dd = ['dataset/train/moire/000', 'dataset/val/moire/000', 'dataset/val/hdrvdp/000']
    model = BayerJDDNetwork().cuda()
    # model = th.load('net7.pth').cpu()
    trainer = Demosaicnet(model)
    # sigma = np.random.uniform(0, 20) / 255
    trainer.train(dd, BATCH_SIZE, EPOCHS)
    trainer.save_model(8)
    # testdata = DemosaicDataset("dataset/test/moire/000")
    # testloader = Data.DataLoader(dataset=testdata, batch_size=BATCH_SIZE, shuffle=True)
    # trainer.test(testloader)







