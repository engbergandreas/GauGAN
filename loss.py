import torch
from torch import nn
import torch.nn.functional as F
from models import VGG19


#discriminator KL divergence -> learning the mean and variance predicted by the encoder

# gen loss = g_loss + kl_loss + vgg_loss + feature_loss
# g loss between - loss between descriminator prediction, and actual label
# kl_loss : encoder output mean,variance

class Gen_loss(nn.Module):
    def forward(self, pred):
        return -pred.mean()

#For learning the mean and variance predicted by the encoder, https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
class KLD_loss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# feature_loss: loss between real desc output and fake desc output, essentially counts the number of correct guesses
class FeatureLossDisc(nn.Module):
    def forward(self, real_disc_outputs, fake_disc_outputs):
        loss = 0
        for real_disc_output,fake_disc_output in zip(real_disc_outputs,fake_disc_outputs):
            loss+= F.l1_loss(real_disc_output,fake_disc_output)
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
