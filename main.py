import numpy as np
import os
import torch.utils.data as Data
from torchvision import transforms
from network.data import DemosaicDataset
from network.jdd import BayerJDDNetwork
from network.trainer import Demosaicnet
from PIL import Image


BATCH_SIZE = 4
EPOCHS = 4
if __name__ == '__main__':
    dataset = DemosaicDataset("dataset/train/moire/000")
    model = BayerJDDNetwork()
    trainer = Demosaicnet(model)
    toPIL = transforms.ToPILImage()
    loader = Data.DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(EPOCHS):
        for idx, batch in enumerate(loader):
            gt = batch["I"]
            subkey = ['sigma', 'M']
            data = dict([(key, batch[key]) for key in subkey])
            out = trainer.forward(data)
            res = trainer.backward(out, gt)
            # img1 = toPIL(out[0])
            # img1.show()
            print("loss:{} psnr:{}".format(res["loss"], res["psnr"]))
            # os.system("pause")







