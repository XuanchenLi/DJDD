import torch as th
from .jdd import BayerJDDNetwork
from metrics import *


class Demosaicnet:
    """
    trainer
    """
    def __init__(self, model, lr=1e-4, cuda=th.cuda.is_available()):
        self.model = model
        self.device = "cpu"
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = th.nn.MSELoss()
        self.psnr = PSNR()

    def forward(self, inputs):
        inputs = inputs.to(self.device)
        outputs = self.model(inputs)
        return outputs

    def backward(self, outputs, targets):
        targets = targets.to(self.device)
        loss = self.loss(outputs, targets)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with th.no_grad():
            psnr = self.psnr(th.clamp(outputs, 0, 1), targets)
        return {"loss": loss.item(), "psnr": psnr.item()}
