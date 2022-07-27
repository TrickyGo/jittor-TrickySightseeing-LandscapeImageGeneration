import jittor as jt
from jittor import init
from jittor import nn
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
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
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

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")

opt.batch_size = 2
val_dataloader = ImageDataset(opt.data_path, mode="val").set_attrs(
    batch_size=opt.batch_size,
    shuffle=False,
    num_workers=1,
)

#@jt.single_process_scope()
def transfer_eval(epoch, sample_numbers=0):
    edit_idx = 0 #0:moutain, 1:sky, 2:water, 3:sea
    os.makedirs(f"{opt.output_path}/images/style_transfer/epoch_{epoch}/editIdx_{edit_idx}", exist_ok=True)
    for i, (ref_B, real_A, photo_id) in enumerate(val_dataloader):
        print("transfering:",i," of ", sample_numbers)
        if i > sample_numbers: return

        fake_B, (image_for_editing, patterns_for_editing) = generator(real_A, ref_B)
        img_sample = np.concatenate([ref_B.data, fake_B.data], -2)
        img = save_image(img_sample, f"{opt.output_path}/images/style_transfer/epoch_{epoch}/editIdx_{edit_idx}/{i}_sample.png", nrow=1)

        pattern_code = patterns_for_editing[edit_idx].data
        #edit pattern_code
        patterns_edited = patterns_for_editing.copy()
        for idx in range(1, opt.batch_size): #copy previous sample's patten code
            patterns_edited[edit_idx][idx] = pattern_code[idx-1]

        fake_B_edit, _ = generator(real_A, image=None, patterns_for_editing=patterns_edited)
        img_sample_edit = np.concatenate([ref_B.data, fake_B_edit.data], -2)
        img_edit = save_image(img_sample_edit, f"{opt.output_path}/images/style_transfer/epoch_{epoch}/editIdx_{edit_idx}/{i}_sample_edit.png", nrow=1)     




sample_numbers = 20
epoch = opt.epoch
transfer_eval(epoch, sample_numbers=sample_numbers)
