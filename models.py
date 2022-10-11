from multiprocessing.context import set_spawning_popen
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import torchvision
import settings

#LATENT_DIM = 256
#NDF = 48
#NUM_CLASSES = 32

#EncoderBlock inspired from https://github.com/kvsnoufal/GauGanPytorch/
class EncoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,with_norm=True):
        super(EncoderBlock,self).__init__()
        kw = 3 #kernel width
        pw = int(np.ceil((kw - 1.0) / 2)) #padding should be half the size of kernel

        if with_norm: #TODO add spectral normalization, remove bias on conv2d as it has no effect after normalization 
            self.block = nn.Sequential(
                                        spectral_norm(nn.Conv2d(in_channels=in_channels,out_channels=out_channels,\
                                            kernel_size=kw,stride=2,padding=pw, bias=False)),           
                                        nn.InstanceNorm2d(out_channels),
                                        nn.LeakyReLU(0.2)
                                        )
        else: #TODO remove as all layers has normalization?
            self.block = nn.Sequential(
                                        nn.Conv2d(in_channels=in_channels,out_channels=out_channels,\
                                                    kernel_size=kw,stride=2,padding=pw),
                                        nn.LeakyReLU(0.2)
                                        )
    def forward(self,x):
        return self.block(x)

#Figure 14 https://arxiv.org/pdf/1903.07291.pdf
class Encoder(nn.Module) : 
    def __init__(self) :
        super().__init__()
        ndf = settings.NDF #number of features, (scales the number of channels in the encoder)
                
        #self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        #self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))

        self.block1 = EncoderBlock(in_channels=3, out_channels=ndf)
        self.block2 = EncoderBlock(in_channels=ndf * 1, out_channels=ndf * 2)
        self.block3 = EncoderBlock(in_channels=ndf * 2, out_channels=ndf * 4)
        self.block4 = EncoderBlock(in_channels=ndf * 4, out_channels=ndf * 8)
        self.block5 = EncoderBlock(in_channels=ndf * 8, out_channels=ndf * 8)
        self.block6 = EncoderBlock(in_channels=ndf * 8, out_channels=ndf * 8)


        self.linear_mu = nn.Linear(in_features=ndf * 8 * 4 * 4, out_features=settings.LATENT_DIM)
        self.linear_var = nn.Linear(in_features=ndf * 8 * 4 * 4, out_features=settings.LATENT_DIM)

    def forward(self, x) : 
        #print(np.shape(x))

        # leaky = nn.LeakyReLU(0.02)
        x = self.block1(x)
        #print("block1", np.shape(x))
        x = self.block2(x)
        #print("block2", np.shape(x))
        x = self.block3((x))
        #print("block3", np.shape(x))
        x = self.block4((x))
        #print("block4", np.shape(x))
        x = self.block5((x))
        #print("block5", np.shape(x))
        x = self.block6((x))
        #print("block6", np.shape(x))

        #print(x.shape)
        x = x.view(x.shape[0], -1) #flattens each bach layer
        #x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        #print("reshaped", np.shape(x))


        mu = self.linear_mu(x)
        var = self.linear_var(x)

        return mu, var

    def compute_latent_vec(self, mu, logvar):
        std = torch.exp(logvar * 0.5) #sort of sqrt variance (?)
        epsilon = torch.rand_like(std)

        return std * epsilon + mu

        # print(np.shape(x))

# Creates SPADE normalization layer based on the given configuration
# SPADE consists of two steps. First, it normalizes the activations using
# your favorite normalization method, such as Batch Norm or Instance Norm.
# Second, it applies scale and bias to the normalized output, conditioned on
# the segmentation map.
# The format of |config_text| is spade(norm)(ks), where
# (norm) specifies the type of parameter-free normalization.
#       (e.g. syncbatch, batch, instance)
# (ks) specifies the size of kernel in the SPADE module (e.g. 3x3)
# Example |config_text| will be spadesyncbatch3x3, or spadeinstance5x5.
# Also, the other arguments are
# |norm_nc|: the #channels of the normalized activations, hence the output dim of SPADE
# |label_nc|: the #channels of the input semantic map, hence the input dim of SPADE

#Figure 10 https://arxiv.org/pdf/1903.07291.pdf
class SPADE(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        nHiddenChannels = 128

        self.bn = nn.BatchNorm2d(num_channels, affine=False)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=settings.NUM_CLASSES, out_channels=nHiddenChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv_gamma = nn.Conv2d(in_channels=nHiddenChannels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv_beta = nn.Conv2d(in_channels=nHiddenChannels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, segmap):
        normalized = self.bn(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        activation = self.block1(segmap)
        gamma = self.conv_gamma(activation)
        beta = self.conv_beta(activation)

        out = normalized * (1 + gamma) + beta #TODO (1 + gamma)

        return out

#Figure 11 https://arxiv.org/pdf/1903.07291.pdf
class SPADEResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = (in_channels != out_channels)
        #TODO create feature middle (?) ie. fmiddle = min(fin, fout) does not seem necessary 
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)#TODO this would be fmiddle as out
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1)#TODO this would be fmiddle as in
        if self.shortcut:
            self.conv_skip = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False)
            #TODO is actually a conv2d(in, out, ks = 1, bias=False)

        #Applys spectral normalization to all convolutional layers
        self.conv1 = spectral_norm(self.conv1)
        self.conv2 = spectral_norm(self.conv2)
        if self.shortcut:
            self.conv_skip = spectral_norm(self.conv_skip)
        #Normalization layers
        self.spade1 = SPADE(num_channels=in_channels)
        self.spade2 = SPADE(num_channels=out_channels)#TODO this would be fmiddle 

        if self.shortcut:
            self.spade_skip = SPADE(num_channels=in_channels)


    def forward(self, x, segmap):
        x_skip = self.compute_shortcut(x, segmap)

        #print(x)
        #print("SPADEresnetblock", np.shape(x))
        x = self.spade1(x, segmap)
        #print("SPADEresnetblock after", np.shape(x))
        #print(self.spade1)

        x = F.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = self.spade2(x, segmap)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)

        #print('x',x.size(),'xs', x_skip.size())
        out = x + x_skip
        return out
        
    def compute_shortcut(self, x, segmap):
        if self.shortcut:
            x_s = self.spade_skip(x, segmap)
            x_s = F.leaky_relu(x_s, 0.2)#TODO remove? in paper this is here
            x_s = self.conv_skip(x_s)
        else:
            x_s = x
        return x_s

#Figure 12 https://arxiv.org/pdf/1903.07291.pdf
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        nf = settings.NDF #scales on 

        self.fc = nn.Linear(settings.LATENT_DIM, 16 * nf * 4 * 4)
        self.upsample = nn.Upsample(scale_factor=2)

        self.block1 = SPADEResnetBlock(16 * nf, 16 * nf)
        self.block2 = SPADEResnetBlock(16 * nf, 16 * nf)
        self.block3 = SPADEResnetBlock(16 * nf, 16 * nf)
        self.block4 = SPADEResnetBlock(16 * nf, 8 * nf)
        self.block5 = SPADEResnetBlock(8 * nf, 4 * nf)
        self.block6 = SPADEResnetBlock(4 * nf, 2 * nf)
        self.block7 = SPADEResnetBlock(2 * nf, 1 * nf)

        self.conv = nn.Conv2d(1 * nf, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, latent_vec, segmap):
        x = self.fc(latent_vec)
        x = x.view(-1, 16 * settings.NDF, 4, 4) #???
        #print(np.shape(x))

        x = self.block1(x, segmap)
        x = self.upsample(x) 
        x = self.block2(x, segmap)

        x = self.upsample(x) #TODO to be removed? in paper this is here
        
        x = self.block3(x, segmap)
        x = self.upsample(x)
        x = self.block4(x, segmap)
        x = self.upsample(x)
        x = self.block5(x, segmap)
        x = self.upsample(x)
        x = self.block6(x, segmap)
        x = self.upsample(x)
        x = self.block7(x, segmap)
        #x = self.upsample(x)
        #print("last upsample size of img is now", np.shape(x))

        x = F.leaky_relu(x, 0.2)
        x = self.conv(x)
        x = torch.tanh(x)

        return x

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalization=True, stride=2):
        super().__init__()
        #TODO set bias to false in conv2d see reason in encoder
        if normalization:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=2, bias=False)), #TODO they use padding = 2, bias = false?
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            self.block = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=2)), #TODO they use padding = 2, bias = false?
                nn.LeakyReLU(0.2),
            )
    def forward(self, x):
        return self.block(x)

#Figure 13 https://arxiv.org/pdf/1903.07291.pdf
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DiscriminatorBlock(3 + settings.NUM_CLASSES, 64, False)
        self.block2 = DiscriminatorBlock(64, 128)
        self.block3 = DiscriminatorBlock(128, 256)
        self.block4 = DiscriminatorBlock(256, 512, stride=1)
        self.block5 = spectral_norm(nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=2)) #TODO padding = 2?
        #DiscriminatorBlock(512, 1, False)

    def forward(self, x): 
        #concat
        #concat_img = torch.concat([x, segmap], 1)
        #print('cocant img', np.shape(concat_img))
        d1 = self.block1(x)
        #print('first',np.shape(res))
        d2 = self.block2(d1)
        #print('sec',np.shape(res))
        d3 = self.block3(d2)
        #print('3',np.shape(res))
        d4 = self.block4(d3)
        #print('4',np.shape(res))
        d5 = self.block5(d4) 
       # print('5',np.shape(res))
        #Maybe add leaky relu here as last layer, even though it is not stated in the paper
        return [d1, d2, d3, d4, d5] #TODO maybe return an array from each output of the different blocks, 


# VGG architecter, used for the perceptual loss using a pretrained VGG network, source: https://github.com/NVlabs/SPADE
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights='IMAGENET1K_V1').features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
