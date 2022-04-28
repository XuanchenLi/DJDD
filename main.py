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


BATCH_SIZE = 64
EPOCHS = 20

if __name__ == '__main__':
    dir = "dataset/train/moire/000"
    pt = os.listdir(dir)
    dd = []
    dd.append(dir)
    """
    for f in pt:
      dd.append(os.path.join(dir, f, 'images'))
    """
    model = BayerJDDNetwork()
    model.cuda()
    # t_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(t_num)
    trainer = Demosaicnet(model)
    """
    for s in dd:
      dataset = DemosaicDataset(s)
      loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    """
    trainer.train(dd, BATCH_SIZE, EPOCHS)
    trainer.save_model()
    testdata = DemosaicDataset("dataset/test/moire/000")
    testloader = Data.DataLoader(dataset=testdata, batch_size=BATCH_SIZE, shuffle=False)
    trainer.test(testloader)







