# from matplotlib.style import use
import jittor as jt
from jittor import init
from jittor import nn
from spectral_norm import spectral_norm
from math import sqrt
import numpy as np

class Generator(nn.Module):
    def __init__(self, nf=64, sem_nc=29):
        super(Generator, self).__init__()

        self.nf = nf
        # self.noise_shape = 256
        # self.noise_mapping_fc = nn.Linear(self.noise_shape, 16 * nf * 12 * 16)

        self.init_conv = nn.Conv2d(sem_nc , 16 * nf, 3, padding=1)
        self.PatternEncoder = PatternEncoder(3, sem_nc)

        self.b0 = ResBlock(16 * nf, 16 * nf, sem_feature_nc=sem_nc, use_attention=False)

        self.b1 = ResBlock(16 * nf, 16 * nf, sem_feature_nc=sem_nc, use_attention=False)
        self.b2 = ResBlock(16 * nf, 16 * nf, sem_feature_nc=sem_nc, use_attention=False)

        self.b3 = ResBlock(16 * nf, 8 * nf, sem_feature_nc=sem_nc, use_attention=False)
        self.b4 = ResBlock(8 * nf, 4 * nf, sem_feature_nc=sem_nc, use_attention=False)
        self.b5 = ResBlock(4 * nf, 2 * nf, sem_feature_nc=sem_nc, use_attention=False)
        self.b6 = ResBlock(2 * nf, 1 * nf, sem_feature_nc=sem_nc, use_attention=False)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2) # mode='bilinear'


    def execute(self, sem, image, patterns_for_editing=None):

        patterns, (image_for_editing, patterns_for_editing) = self.PatternEncoder(sem, image, patterns_for_editing=patterns_for_editing)

        # (image_for_editing, patterns_for_editing) = (None, None)
        # noise = jt.randn(sem.size(0), self.noise_shape)

        sem_feature_pyramid = [
            nn.interpolate(sem, size=(12, 16)),
            nn.interpolate(sem, size=(24, 32)),
            nn.interpolate(sem, size=(48, 64)),
            nn.interpolate(sem, size=(96, 128)),
            nn.interpolate(sem, size=(192, 256)),
            nn.interpolate(sem, size=(384, 512))
        ]

        # x = self.noise_mapping_fc(noise)
        # x = x.view(-1, 16*self.nf, 12, 16)
        x = self.init_conv(patterns)

        x = self.b0(x, sem_feature_pyramid[0])
        x = self.up(x)

        x = self.b1(x, sem_feature_pyramid[1])

        x = self.b2(x, sem_feature_pyramid[1])
        x = self.up(x)

        x = self.b3(x, sem_feature_pyramid[2])
        x = self.up(x)

        x = self.b4(x, sem_feature_pyramid[3])
        x = self.up(x)

        x = self.b5(x, sem_feature_pyramid[4])
        x = self.up(x)

        x = self.b6(x, sem_feature_pyramid[5])
        x = self.conv_img(nn.leaky_relu(x, 2e-1))
        x = nn.Tanh()(x)

        return x, (image_for_editing, patterns_for_editing)


class PatternEncoder(nn.Module):        
    #Semantic Region Pattern identifier
    ## Image -> 29 parts -> 29 Channels pattern code
    def __init__(self, input_nc, output_nc):
        super(PatternEncoder, self).__init__()

        def downsampling_res_block(in_filters, out_filters, kernel_size=4, stride=2, padding=1, normalization=True):
            layers = [nn.Conv(in_filters, out_filters, kernel_size, stride=stride, padding=padding)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters, eps=1e-05, momentum=0.1, affine=True))
            layers.append(nn.LeakyReLU(scale=0.2))
            return layers

        self.shared_pattern_encoder = nn.Sequential(
                                                    *downsampling_res_block(3, 32, kernel_size=3, normalization=False),
                                                    *downsampling_res_block(32, 32),                                               
                                                    nn.Conv(32, 1, 4, padding=1, bias=False),
                                                    nn.AdaptiveAvgPool2d((12,16)),
                                                    )
        # num_classes = 10     
        style_code_dim = 1                                       
        self.fc = nn.Linear(12*16, style_code_dim)


    def softargmax(self, x, beta=1e2):
        x_range = jt.arange(x.size(-1), dtype=x.dtype)
        # print(x.shape,x_range,(nn.softmax(x*beta) * x_range).shape)
        # print(jt.sum(nn.softmax(x*beta) * x_range, dim=-1).shape)
        return jt.sum(nn.softmax(x*beta) * x_range, dim=-1)
        # y = tf.reduce_sum(tf.cumsum(tf.ones_like(x)) * tf.exp(beta * x) / tf.reduce_sum(tf.exp(beta * x))) - 1


    def execute(self, sem, image, patterns_for_editing=None):
        if patterns_for_editing is not None:
            extract_patterns = False
        else:
            extract_patterns = True
            patterns_for_editing = []

        #     image = jt.randint(0, 255, shape=(sem.size(0), 3, sem.size(2), sem.size(3)))
        for sem_idx in range(sem.size(1)):
            if extract_patterns == False: #editing
                pattern = patterns_for_editing[sem_idx]
            else:#training or testing
                cur_sem = sem[:, sem_idx:sem_idx+1, :, :]
                cur_image_part = jt.multiply(image, cur_sem.repeat(1,3,1,1))
                x = self.shared_pattern_encoder(cur_image_part) 
                x = jt.reshape(x, (x.shape[0], (- 1)))
                x = self.fc(x)
                # pattern = jt.argmax(x,-1, keepdims=True)[0] ##TODO:No gradient????
                # approximated argmax:
                # pattern = self.softargmax(x)
                pattern = x
                patterns_for_editing.append(pattern)

            # pattern = jt.unsqueeze(pattern, -1)
            pattern = jt.unsqueeze(pattern, -1)
            pattern = jt.unsqueeze(pattern, -1)
            pattern = jt.expand(pattern, (sem.shape[0], 1, 12, 16))

            if sem_idx == 0:
                patterns = pattern
            else:
                patterns = jt.concat([patterns, pattern], dim=1)
            # print(patterns);assert 0
        return patterns, (image, patterns_for_editing)


class ResBlock(nn.Module):
    
    def __init__(self, fin, fout, sem_feature_nc=32, use_attention=False):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # create conv layers
        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        contain_spectral_norm = True
        if contain_spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)
        
        self.SSI_0 = SSI(fin, sem_feature_nc, use_attention=use_attention)

        self.SSI_1 = SSI(fout, sem_feature_nc, use_attention=use_attention)

        if self.learned_shortcut:
            self.SSI_s = SSI(fin, sem_feature_nc, use_attention=use_attention)



    def execute(self, x, sem_feature):
        # print("in ResBlock:",x.shape, sem_feature.shape)
        x_s = self.shortcut(x, sem_feature)
        dx = self.SSI_0(x, sem_feature)

        dx = self.conv_0(self.actvn(dx))

        dx = self.SSI_1(dx, sem_feature)

        dx = self.conv_1(self.actvn(dx))

        out = x_s + dx
        return out

    def shortcut(self, x, sem_feature):
        if self.learned_shortcut:
            x_s = self.SSI_s(x, sem_feature)
            x_s = self.conv_s(x_s)
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return nn.leaky_relu(x, 2e-1)


class SSI(nn.Module): # Spatial Style Injection
    def __init__(self, norm_nc, sem_feature_nc, use_attention=False):
        super().__init__()

        self.use_attention = use_attention
        self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        nhidden = 256
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(sem_feature_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        #print("sem_feature_nc",sem_feature_nc)
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def execute(self, x, sem_feature):
        out = x

        # sem
        actv = self.mlp_shared(sem_feature)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        # print("actv shape",actv.shape, gamma.shape)

        normalized = self.param_free_norm(out)
        out = normalized * (1 + gamma) + beta


        return out



def start_grad(model):
    for param in model.parameters():
        param.start_grad()

def stop_grad(model):
    for param in model.parameters():
        param.stop_grad()

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        jt.init.gauss_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        jt.init.gauss_(m.weight, 1.0, 0.02)
        jt.init.constant_(m.bias, 0.0)


class Discriminator(nn.Module):

    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        num_D = 2
        self.model = nn.Sequential()
        for _ in range(num_D):
            self.model.append(MultiscalePatchDiscriminator(in_channels))
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
    
    def execute(self, input):
        result = []
        for D in self.model:
            out = D(input)
            result.append(out)
            input = self.avgpool(input)
        return result


class MultiscalePatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3+29):
        super(MultiscalePatchDiscriminator, self).__init__()

        kw = 4
        padw = 2 #int(np.ceil((kw - 1.0) / 2))
        nf = 64
        layer_num = 4
        sequence = [[nn.Conv2d(in_channels, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2)]]

        for n in range(1, layer_num):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == layer_num - 1 else 2
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw),
                          nn.BatchNorm2d(nf),
                          nn.LeakyReLU(0.2)]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        self.model = nn.Sequential()
        for n in range(len(sequence)):
            self.model.append(nn.Sequential(*sequence[n]))
        
        for m in self.modules():
            weights_init_normal(m)

    def execute(self, input):
        results = [input]
        for submodel in self.model:
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)
        return results[1:]
