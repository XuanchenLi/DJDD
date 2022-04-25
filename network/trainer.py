from .metrics import *


class Demosaicnet:
    """
    trainer
    """
    def __init__(self, model=None, lr=1e-4, cuda=th.cuda.is_available()):
        self.model = model
        self.device = "cpu"
        if cuda:
            self.device = "cuda"
        # self.model.to(self.device)
        self.opt = th.optim.Adam(self.model.parameters(), lr=lr)
        self.loss = th.nn.MSELoss()
        self.psnr = PSNR()
        self.testloader = None
        self.trainloader = None
        self.epochs = 64

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
        return {"loss": loss.item(), "psnr": psnr.item()}

    def train(self, dataloader, epochs):
        self.trainloader = dataloader
        self.epochs = epochs
        self.model.train()
        for epoch in range(self.epochs):
            for idx, batch in enumerate(self.trainloader):
                gt = batch["I"]
                subkey = ['sigma', 'M']
                data = dict([(key, batch[key]) for key in subkey])
                out = self.forward(data)
                res = self.backward(out, gt)
                # img1 = toPIL(out[0])
                # img1.show()
                print("train loss:{} psnr:{}".format(res["loss"], res["psnr"]))
                # os.system("pause")

    def test(self, dataloader):
        self.testloader = dataloader
        self.model.eval()
        with th.no_grad():
            for idx, batch in enumerate(self.testloader):
                gt = batch["I"]
                subkey = ['sigma', 'M']
                data = dict([(key, batch[key]) for key in subkey])
                self.opt.zero_grad()
                out = self.forward(data)
                loss = self.loss(out, gt)
                psnr = self.psnr(th.clamp(out, 0, 1), out)
                # img1 = toPIL(out[0])
                # img1.show()
                print("test loss:{} psnr:{}".format(loss.item(), psnr))
                # os.system("pause")

    def save_model(self):
        th.save(self.model, 'net.pth')

    def load_model(self, path):
        self.model = th.load(path)
