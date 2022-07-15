import glob
import random
import os
import numpy as np
# from diffAug import rand_brightness, rand_contrast, rand_saturation
import jittor as jt
from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image
import pickle

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms=None, one_hot_label=True):
        super().__init__()

        transforms = [
            transform.Resize(size=(384, 512), mode=Image.BICUBIC),
            transform.ToTensor(),
            transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]
        self.transforms = transform.Compose(transforms)
        self.mode = mode
        if self.mode == 'train':
            self.files = sorted(glob.glob(os.path.join(root, "train", "imgs") + "/*.*"))

        self.labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

        self.one_hot_label = one_hot_label
        if self.one_hot_label:
            self.transforms_label = transform.Compose([
                                transform.Resize(size=(384, 512), mode=Image.NEAREST),
                                transform.ToTensor(),
                              ])

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))
        
        if self.mode == "train":
            img_A = Image.open(self.files[index % len(self.files)])
            if np.random.random() < 0.5:
                img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
                img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
            img_A = self.transforms(img_A)
        else:
            img_A = np.empty([1])

        if self.one_hot_label:
            img_B = self.transforms_label(img_B)[0:1, :,:]*255
            _, h, w = img_B.shape
            nc = 29
            input_label = jt.zeros((nc, h, w)).detach()
            src = jt.ones((nc, h, w)).detach()
            img_B = input_label.scatter_(0, img_B, src).detach()
            del input_label,src


        return img_A, img_B, photo_id
        #img_A = image, img_B = label


def DiffAug(fake_A, img_A, img_B):
    #color transformations
    fake_A, img_A = rand_brightness(fake_A), rand_brightness(img_A)
    fake_A, img_A = rand_saturation(fake_A), rand_saturation(img_A)
    fake_A, img_A = rand_contrast(fake_A),   rand_contrast(img_A)
    fake_A, img_A = add_noise(fake_A), add_noise(img_A)
    
    #scale transformations
    #fake_A, img_A, img_B = rand_upscale(fake_A, img_A, img_B)

    return fake_A, img_A, img_B

def rand_brightness(x, mul_factor=0.5):
    x = x + (jt.rand(x.shape[0], 1, 1, 1, dtype=x.dtype) - 0.5) * mul_factor
    return x


def rand_saturation(x, mul_factor=0.5):
    x_mean = x.mean(dim=1, keepdims=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype) * 2) * mul_factor + x_mean
    return x


def rand_contrast(x, mul_factor=0.5):
    x_mean = x.mean(dims=[1, 2, 3], keepdims=True)
    x = (x - x_mean) * (jt.rand(x.size(0), 1, 1, 1, dtype=x.dtype) + 0.5) * mul_factor + x_mean
    return x

def add_noise(x, noise_std=0.1,noise=0, mul_factor=0.5):
    batch_size = x.shape[0]
    sigma = jt.randn([batch_size, 1, 1, 1]).abs() * noise_std
    sigma = jt.where(jt.rand([batch_size, 1, 1, 1]) < noise * mul_factor, sigma, jt.zeros_like(sigma))
    x = x + jt.randn([batch_size, x.shape[1], x.shape[2], x.shape[3]]) * sigma
    return x


def rand_upscale(fake_x, x, x_sem, ratio=0.25):
    up_ratio = 1.0 + ratio * random.random()
    orig_h, orig_w = x.size(2),x.size(3)

    fake_x = jt.nn.interpolate(fake_x, scale_factor=up_ratio, mode='bilinear')
    x = jt.nn.interpolate(x, scale_factor=up_ratio, mode='bilinear')
    x_sem = jt.nn.interpolate(x_sem, scale_factor=up_ratio, mode='nearest')

    return center_crop(fake_x, orig_h, orig_w), center_crop(x, orig_h, orig_w), center_crop(x_sem, orig_h, orig_w)

def center_crop(x, orig_h, orig_w):
    h, w = x.size(2), x.size(3)
    x1 = int(round((h - orig_h) / 2.))
    y1 = int(round((w - orig_w) / 2.))
    return x[:, :, x1:x1+orig_h, y1:y1+orig_w]