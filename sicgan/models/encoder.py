import torch
import torch.nn as nn
from sicgan.models.heads import vae_encoder_head
from torch.autograd import Variable

class encoder_head(nn.Module):
    def __init__(self,cfg):
        super(encoder_head,self).__init__()

        self.vae_encoder = vae_encoder_head(cfg)

    def sample_z(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std) + mu

    def forward(self,imgs):
        means,sigmas = self.vae_encoder(imgs)
        z_x = self.sample_z(means,sigmas)
        return z_x,means,sigmas