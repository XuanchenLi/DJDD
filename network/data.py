import numpy as np
import os
from PIL import Image, ImageChops
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from .mosaic import *


class DemosaicDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.path = os.listdir(self.root_dir)

    def __getitem__(self, index):
        sample_name = self.path[index]
        sample_url = os.path.join(self.root_dir, sample_name)
        sample = Image.open(sample_url)
        sample = sample.resize((int(sample.width*0.5), int(sample.height*0.5)), Image.BICUBIC)
        sample = self.augment(sample)
        sample = th.from_numpy(np.transpose(sample, (2, 0, 1)).copy()).float()
        m_sample = bayer(sample)
        # toPIL = transforms.ToPILImage()
        # img = toPIL(sample)
        # img.show()
        sigma = np.random.uniform(0, 20)
        transform = AddGaussianNoise(0, sigma**2)
        m_sample = transform(m_sample).float()

        return {"sigma": sigma, "M": m_sample, "I": sample}

    def __len__(self):
        return len(self.path)

    def augment(self, img):
        img = img.rotate(90 * np.random.randint(0, 4))
        if np.random.randint(0, 2):
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        num = np.random.randint(0, 5)
        if num == 1:
            img = ImageChops.offset(img, 1, 0)
        elif num == 2:
            img = ImageChops.offset(img, -1, 0)
        elif num == 3:
            img = ImageChops.offset(img, 0, 1)
        elif num == 4:
            img = ImageChops.offset(img, 0, -1)
        return img


class AddGaussianNoise(object):
    def __init__(self, mean=0.0, variance=1.0):
        self.mean = mean
        self.variance = variance

    def __call__(self, img):
        image = np.array(img/255, dtype=float)
        noise = np.random.normal(self.mean, self.variance ** 0.5, image.shape)
        out = img + noise
        if out.min() < 0:
            low_clip = -1
        else:
            low_clip = 0
        out = np.clip(out, low_clip, 1.0)
        out = out*255
        return out

