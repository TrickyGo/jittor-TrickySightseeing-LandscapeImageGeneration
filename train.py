# from regex import F
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

from tensorboardX import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

jt.flags.use_cuda = 1

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--data_path", type=str, default="./jittor_landscape_200k")
parser.add_argument("--output_path", type=str, default="./results/flickr")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=384, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=3, help="interval between model checkpoints")
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
os.makedirs(f"{opt.output_path}/saved_models/", exist_ok=True)

writer = SummaryWriter(opt.output_path)

# Loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_pixelwise = nn.L1Loss()

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

# Initialize generator and discriminator
sem_nc = 29
generator = Generator(sem_nc=sem_nc)
discriminator = Discriminator(in_channels=3+sem_nc)

if opt.epoch != 0:
    # Load pretrained models
    generator.load(f"{opt.output_path}/saved_models/generator_{opt.epoch}.pkl")
    discriminator.load(f"{opt.output_path}/saved_models/discriminator_{opt.epoch}.pkl")

# Optimizers
optimizer_G = jt.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = jt.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Configure dataloaders
dataloader = ImageDataset(opt.data_path, mode="train").set_attrs(
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = ImageDataset(opt.data_path, mode="val").set_attrs(
    batch_size=10,
    shuffle=False,
    num_workers=1,
)

#@jt.single_process_scope()
def eval(epoch, writer):
    cnt = 1
    os.makedirs(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}", exist_ok=True)
    for i, (ref_B, real_A, photo_id) in enumerate(val_dataloader):
        fake_B, (image_for_editing, patterns_for_editing) = generator(real_A, ref_B)
        
        if i == 0:
            # visual image result
            img_sample = fake_B.data
            img = save_image(img_sample, f"{opt.output_path}/images/epoch_{epoch}_sample.png", nrow=5)
            
            writer.add_image('val/image', img.transpose(2,0,1), epoch)


        fake_B = ((fake_B + 1) / 2 * 255).numpy().astype('uint8')
        for idx in range(fake_B.shape[0]):
            cv2.imwrite(f"{opt.output_path}/images/test_fake_imgs/epoch_{epoch}/{photo_id[idx]}.jpg", fake_B[idx].transpose(1,2,0)[:,:,::-1])
            cnt += 1

warmup_times = -1
run_times = 3000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------
# Perceptual loss that uses a pretrained precetion network
# from jittor.models import vgg19
from inception import Inception3
class PerceptionLoss(nn.Module):
    def __init__(self):
        super(PerceptionLoss, self).__init__()
        self.percetion_net = Inception3()
        self.percetion_net.load("jittorhub://inception_v3.pkl")
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 128, 1.0 / 128, 1.0 / 128, 1.0 / 128, 1.0 / 128, 1.0 / 128,
        #                 1.0 / 64 , 1.0 / 64 , 1.0 / 64 , 1.0 / 64 , 1.0 / 64 , 1.0 / 64 ,
        #                 1.0 / 32 , 1.0 / 32 , 1.0 / 16,  1.0 / 8,   1.0 / 4  , 1.0]
        self.weights = [1.0 / 8,  1.0 / 4, 1.0]

    def execute(self, x, y):
        _, x_feat = self.percetion_net(x)
        _, y_feat = self.percetion_net(y)
        # print(len(x_feat))

        loss = 0
        for i in range(len(x_feat)):
            loss +=  self.weights[i] * self.criterion(x_feat[i], y_feat[i])
        return loss



prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, (real_B, real_A, _) in enumerate(dataloader):
                
        # Adversarial ground truths
        valid = jt.ones([real_A.shape[0], 1]).stop_grad()
        fake = jt.zeros([real_A.shape[0], 1]).stop_grad()
        fake_B, _ = generator(real_A, real_B)

        if epoch > 600:
            apply_diffaug = True
        else:
            apply_diffaug = False  
        if apply_diffaug:     
            #apply DiffAug
            fake_B, real_B, real_A = DiffAug(fake_B, real_B, real_A)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        start_grad(discriminator)
        fake_AB = jt.contrib.concat((real_A, fake_B), 1) 
        output_D_fake = discriminator(fake_AB.detach())
        loss_D_fake = 0
        for pred_fake in output_D_fake:
            pred = pred_fake[-1]
            loss_D_fake += criterion_GAN(pred, False) / len(output_D_fake)


        real_AB = jt.contrib.concat((real_A, real_B), 1)
        output_D_real = discriminator(real_AB.detach())
        loss_D_real = 0
        for pred_real in output_D_real:
            pred = pred_real[-1]
            loss_D_real += criterion_GAN(pred, True) / len(output_D_real)

        loss_D = (loss_D_fake + loss_D_real) * 0.5
        optimizer_D.step(loss_D)
        # writer.add_scalar('train/loss_D', loss_D.item(), epoch * len(dataloader) + i)

        # ------------------
        #  Train Generators
        # ------------------
        stop_grad(discriminator)        
        fake_AB = jt.contrib.concat((real_A, fake_B), 1) 
        output_D_fake = discriminator(fake_AB)
        loss_G_GAN = 0
        for pred_fake in output_D_fake:
            pred = pred_fake[-1]
            loss_G_GAN += criterion_GAN(pred, True) / len(output_D_fake)
        
        loss_G_L1 = criterion_pixelwise(fake_B, real_B)

        is_decaying_L1 = False
        if is_decaying_L1:
            # decaying L1 loss from [init_L1] to 0 within the first [decaying_period] epochs
            decaying_period = 200
            init_L1 = 10
            lambda_pixel = max(0, (decaying_period-epoch+opt.epoch)/decaying_period) * init_L1
        else:
            lambda_pixel = 0

        loss_G = loss_G_GAN + lambda_pixel * loss_G_L1

        w_loss_G_feat = True
        if w_loss_G_feat:
            lambda_feat = 30
            num_D = len(pred_fake)
            GAN_Feat_loss = 0
            for d in range(num_D):  # for each discriminator
                num_intermediate_outputs = len(pred_fake[d]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = criterion_pixelwise(
                        pred_fake[d][j], pred_real[d][j].detach())
                    GAN_Feat_loss += unweighted_loss * lambda_feat / num_D
            loss_G_feat = GAN_Feat_loss
            loss_G += lambda_feat * loss_G_feat
        else:
            loss_G_feat = 0

        if epoch > 300:
            w_loss_G_perc = True
        else:
            w_loss_G_perc = False
        if w_loss_G_perc:
            criterionPerc = PerceptionLoss()
            lambda_perc = 5.0
            loss_G_perc = criterionPerc(fake_B, real_B) * lambda_perc
        else:
            loss_G_perc = jt.zeros(1)


        optimizer_G.step(loss_G)
        # writer.add_scalar('train/loss_G', loss_G.item(), epoch * len(dataloader) + i)

        jt.sync_all(True)

        if jt.rank == 0:
            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            jt.sync_all()
            if batches_done % 5 == 0:
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f, feat: %f, perc: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D.numpy()[0],
                        loss_G.numpy()[0],
                        loss_G_L1.numpy()[0],
                        loss_G_GAN.numpy()[0],
                        loss_G_feat.numpy()[0],
                        loss_G_perc.numpy()[0],
                        time_left,
                    )   
                )
        
        is_debug_mode = False
        if is_debug_mode:
            if i > 10:
                eval(epoch, writer)
                assert 0

    if jt.rank == 0 and opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        eval(epoch, writer)
        # Save model checkpoints
        generator.save(os.path.join(f"{opt.output_path}/saved_models/generator_{epoch}.pkl"))
        discriminator.save(os.path.join(f"{opt.output_path}/saved_models/discriminator_{epoch}.pkl"))