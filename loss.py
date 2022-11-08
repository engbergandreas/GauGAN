import torch
from torch import nn
import torch.nn.functional as F
from models import VGG19
from torch import cuda


#discriminator KL divergence -> learning the mean and variance predicted by the encoder

# gen loss = g_loss + kl_loss + vgg_loss + feature_loss
# g loss between - loss between descriminator prediction, and actual label
# kl_loss : encoder output mean,variance

class Gen_loss(nn.Module): 
    def forward(self, pred):
        return -torch.mean(pred)

#For learning the mean and variance predicted by the encoder, https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
class KLD_loss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# feature_loss: loss between real desc output and fake desc output, 
class FeatureLossDisc(nn.Module):
    def forward(self, real_disc_outputs, fake_disc_outputs):
        loss = 0
        for real_disc_output,fake_disc_output in zip(real_disc_outputs,fake_disc_outputs):
            loss += F.l1_loss(fake_disc_output, real_disc_output.detach())
            
        return loss

# Perceptual loss that uses a pretrained VGG network
# vgg_loss: loss between generated image (by generator) and actual image
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        if torch.cuda.is_available():
            self.vgg = VGG19().cuda()
        else:
            self.vgg = VGG19()
        # self.vgg = VGG19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class HingeLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hingeloss = nn.HingeEmbeddingLoss()
    
    def forward(self, x, real=True):
        if real:
            return self.hingeloss(x, torch.ones_like(x))
        else:
            return self.hingeloss(x, torch.ones_like(x)*-1)


#must chnage depending on if its true of false
class Hinge(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.zero_tensor = None

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor =  torch.cuda.FloatTensor(1).fill_(0) if torch.cuda.is_available() else torch.FloatTensor(1).fill_(0)
            #input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_() if torch.cuda.is_available() else torch.FloatTensor(bs, nc, h, w).zero_()
            
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def forward(self, pred, is_target_real):
        if is_target_real:
            minval = torch.min(pred - 1, self.get_zero_tensor(pred))
            loss = -torch.mean(minval)
        else:
            minval = torch.min(-pred - 1, self.get_zero_tensor(pred))
            loss = -torch.mean(minval)

        return loss