from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage
import os
from network.data import DemosaicDataset
import torch.utils.data as Data
import torch as th
from .metrics import *


class Demosaicnet:
    """
    trainer
    """
    def __init__(self, model=None, lr=1e-4, cuda=th.cuda.is_available(), pretrained=False):
        self.model = model
        self.device = "cpu"
        if cuda:
            self.device = "cuda"
        # self.model.to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-8)
        self.loss = th.nn.MSELoss()
        self.psnr = PSNR()
        self.testloader = None
        self.trainloader = None
        self.epochs = 0

    def forward(self, inputs):
        sigma = inputs["sigma"]
        img = inputs["M"]
        # img = img.to(self.device)
        outputs = self.model(img, sigma)
        return outputs

    def backward(self, outputs, targets):
        # targets = targets.to(self.device)
        loss = self.loss(outputs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with th.no_grad():
            psnr = self.psnr(th.clamp(outputs, 0, 1), targets)
        return {"loss": loss.item() / outputs.shape[0] , "psnr": psnr.item()}

    def train(self, dirs, bs, epochs):
        self.dirs = dirs
        # self.trainloader = dataloader
        self.epochs = epochs
        self.model.train()
        for epoch in range(self.epochs):
            if epoch != 0 and epoch % 10 == 0:
                for params in self.opt.param_groups:
                    params['lr'] *= 0.1
            for s in self.dirs:
                dataset = DemosaicDataset(s)
                self.trainloader = Data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
                for idx, batch in enumerate(self.trainloader):
                    gt = batch["I"]
                    subkey = ['sigma', 'M']
                    data = dict([(key, batch[key]) for key in subkey])
                    out = self.forward(data)
                    res = self.backward(out, gt)
                    # img1 = toPIL(out[0])
                     # img1.show()
                    print("train#{}# loss:{} psnr:{}".format(epoch, res["loss"], res["psnr"]))
                     # os.system("pause")

    def test(self, dataloader):
        self.testloader = dataloader
        self.model.eval()
        toPIL = ToPILImage()
        with th.no_grad():
            for idx, batch in enumerate(self.testloader):
                gt = batch["I"]
                subkey = ['sigma', 'M']
                data = dict([(key, batch[key]) for key in subkey])
                self.opt.zero_grad()
                out = self.forward(data)
                loss = self.loss(out, gt)
                psnr = self.psnr(th.clamp(out, 0, 1), gt)
                img1 = toPIL(out[0])
                img1.show()
                img2 = toPIL(gt[0])
                img2.show()
                print("test loss:{} psnr:{}".format(loss.item() / out.shape[0], psnr))
                os.system("pause")

    def save_model(self):
        th.save(self.model, 'net.pth')

    def load_model(self, path):
        self.model = th.load(path)
