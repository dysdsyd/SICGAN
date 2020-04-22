import torch
import torch.nn as nn
from torch.nn import functional as F

def flatten(x, start_dim=1, end_dim=-1):
    return x.flatten(start_dim=start_dim, end_dim=end_dim)
  
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

class vae_encoder_head(nn.Module):
    def __init__(self,cfg,init_weights=True):
        super(vae_encoder_head,self).__init__()

        # self.in_features = 3
        layers = [
                    #192x256
                    nn.Conv2d(3, 16, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(16, 16, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(16, 32, 3, 2, 1),
                    nn.ReLU(),
                    #96x128
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1),
                    nn.ReLU(),
                    #48x64
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.ReLU(),
                    #24x32
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 128, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(128, 256, 5, 2, 2),
                    nn.ReLU(),
                    #12x16
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(256, 512, 5, 2, 2),
                    nn.ReLU(),
                    #6x8
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.ReLU(),
                    nn.Conv2d(512, 512, 3, 1, 1),
                    nn.ReLU(),
        ]
        self.conv_net = nn.Sequential(*layers)
        self.flat_net = Flatten()
        self.fc_mu = nn.Linear(512*6*8,200)
        self.fc_sigma = nn.Linear(512*6*8,200)
        #Apply Tanh after this to sigma Not sure of this doesn't make sense for sigma to be -ve

        if init_weights:
            self._initialize_weights()

    def forward(self,imgs):
        x = self.conv_net(imgs)
        x = self.flat_net(x)
        means = self.fc_mu(x)
        sigmas = self.fc_sigma(x)
        return means,sigmas

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
