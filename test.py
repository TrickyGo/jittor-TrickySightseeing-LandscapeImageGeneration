import jittor as jt
from jittor import init
from jittor import nn
# import jittor.transform as transform
import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import cv2
import time

from models import *
from datasets import *


import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")

opt = parser.parse_args()
print(opt)

def save_image(img, path, nrow=10):
    N,C,W,H = img.shape
    if (N%nrow!=0):
        print("save_image error: N%nrow!=0")
        return
    img=img.transpose((1,0,2,3))
    ncol=int(N/nrow)
    img2=img.reshape([img.shape[0],-1,H])
    img=img2[:,:W*ncol,:]
    for i in range(1,int(img2.shape[1]/W/ncol)):
        img=np.concatenate([img,img2[:,W*ncol*i:W*ncol*(i+1),:]],axis=2)
    min_=img.min()
    max_=img.max()
    img=(img-min_)/(max_-min_)*255
    img=img.transpose((1,2,0))
    if C==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path,img)
    return img

os.makedirs(f"{opt.output_path}/images/", exist_ok=True)

# Initialize generator and discriminator
sem_nc = 29
generator = Generator(sem_nc=sem_nc)


# Load pretrained models
generator.load(f"{opt.output_path}/saved_models/generator_803.pkl")

val_dataloader = ImageDataset(opt.data_path, mode="val").set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)

#@jt.single_process_scope()
def eval():
    cnt = 1
    os.makedirs(f"{opt.output_path}/", exist_ok=True)
    for i, (ref_B, real_A, photo_id) in enumerate(val_dataloader):
        fake_B, _ = generator(real_A, ref_B)
            
        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1

print("testing:")
eval()