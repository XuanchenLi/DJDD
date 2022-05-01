from PIL import Image
import matplotlib.pyplot as plt
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
    def __init__(self, model=None, lr=1e-6, cuda=th.cuda.is_available(), pretrained=False):
        self.model = model
        self.device = "cpu"
        if cuda:
            self.device = "cuda"
        # self.model.to(self.device)
        if model is not None:
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
        # outputs = self.model(img)
        return outputs

    def backward(self, outputs, targets):
        # targets = targets.to(self.device)
        loss = self.loss(outputs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        psnr_ave = 0
        with th.no_grad():
            for i in range(outputs.shape[0]):
                psnr_ave += self.psnr(th.clamp(outputs[i], 0, 1), targets[i])
        psnr_ave /= outputs.shape[0]
        return {"loss": loss.item() / outputs.shape[0], "psnr": psnr_ave}

    def train(self, dirs, bs, epochs):
        self.dirs = dirs
        # self.sigma = sigma
        # self.trainloader = dataloader
        self.epochs = epochs
        self.model.train()
        for epoch in range(self.epochs):
            if epoch != 0:
                pass
                # self.save_model(13 + epoch * 0.1)
            if epoch != 0 and epoch % 5 == 0:
                for params in self.opt.param_groups:
                    params['lr'] *= 0.1
            for s in self.dirs:
                dataset = DemosaicDataset(s)
                self.trainloader = Data.DataLoader(dataset=dataset, batch_size=bs, shuffle=True)
                for idx, batch in enumerate(self.trainloader):
                    th.cuda.empty_cache()
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
        # self.model.eval()
        toPIL = ToPILImage()
        with th.no_grad():
            for idx, batch in enumerate(self.testloader):
                gt = batch["I"]
                subkey = ['sigma', 'M']
                data = dict([(key, batch[key]) for key in subkey])
                self.opt.zero_grad()
                out = self.forward(data)
                # print(out.shape)
                loss = self.loss(out, gt)
                psnr = self.psnr(out, gt)
                psnr_ave = 0
                for i in range(out.shape[0]):
                    psnr_ave += self.psnr(th.clamp(out[i], 0, 1), gt[i])
                psnr_ave /= out.shape[0]
                plt.figure()
                plt.subplot(1, 2, 1)
                plt.imshow(toPIL(th.clamp(out[0].cpu(), 0, 1)))
                plt.xlabel("ours")
                plt.title("noise_level:{}".format(batch['sigma'][0] * 255))
                plt.subplot(1, 2, 2)
                plt.imshow(toPIL(gt[0].cpu()))
                plt.xlabel("ground-truth")
                plt.show()
                print("test loss:{} psnr:{}".format(loss.item() / out.shape[0], psnr_ave))
                os.system("pause")

    def save_model(self, id):
        th.save(self.model, 'net{}.pth'.format(id))

    def load_model(self, path):
        self.model = th.load(path)
