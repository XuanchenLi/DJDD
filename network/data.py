import numpy as np
import os
from PIL import Image, ImageChops, ImageFile
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import skimage
from .mosaic import *


ImageFile.LOAD_TRUNCATED_IMAGES = True


class DemosaicDataset(Dataset):
    def __init__(self, root_dir):
        """
        self.path = []
        self.root_dir = root_dir
        for f in self.root_dir:
          p = os.listdir(f)
          for pp in p:
            self.path.append(os.path.join(f,pp))
        """
        self.root_dir = root_dir
        self.path = os.listdir(root_dir)
        # self.sigma = sigma

    def __getitem__(self, index):
        sample_name = self.path[index]
        sample_url = os.path.join(self.root_dir, sample_name)
        # sample_url = self.path[index]
        sample = Image.open(sample_url).convert("RGB")
        # toPIL = transforms.ToPILImage()
        # sample = sample.resize((int(sample.width*0.5), int(sample.height*0.5)), Image.BICUBIC)
        sample = self.augment(sample)
        sample = th.from_numpy(np.transpose(sample, (2, 0, 1)).copy()).float() / (2**8-1)
        # img = toPIL(m_sample)
        sigma = np.random.rand() * 20 / 255
        transform = AddGaussianNoise(0, sigma)
        m_sample = transform(sample).float()
        m_sample = bayer(m_sample)
        # img = toPIL(m_sample)
        # img.show()
        # print(m_sample - sample)
        # return {"sigma": sigma, "M": m_sample.cuda(), "I": sample.cuda()}
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
    def __init__(self, mean, sigma):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        return th.from_numpy(skimage.util.random_noise(img, mode='gaussian', var=self.sigma ** 2))

