import numpy as np
import os
import torch.utils.data as Data
from torchvision import transforms
from network.data import DemosaicDataset
from network.jdd import BayerJDDNetwork
from network.trainer import Demosaicnet
from PIL import Image
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
BATCH_SIZE = 64
EPOCHS = 32
if __name__ == '__main__':
    dataset = DemosaicDataset("dataset/train/moire/000")
    model = BayerJDDNetwork()
    # t_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(t_num)
    trainer = Demosaicnet(model)
    # toPIL = transforms.ToPILImage()
    loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    trainer.train(loader, EPOCHS)
    testdata = DemosaicDataset("dataset/test/moire/000")
    testloader = Data.DataLoader(dataset=testdata, batch_size=BATCH_SIZE, shuffle=False)
    trainer.test(testloader)







