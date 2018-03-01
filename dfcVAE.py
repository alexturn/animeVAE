import torch
import torch.nn as nn
import torch.distributions as dist
import torchvision.models as models


from torch.autograd import Variable

magic_constant = Variable(torch.Tensor([1e-6])).cuda()
def init_layers(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.normal_(1, 0.02)


class dfcEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(dfcEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(4,4), 
                      stride=(2,2), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        init_layers(self)

    def forward(self, x):
        return self.block(x)


class dfcDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(dfcDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3,3),
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        init_layers(self)

    def forward(self, x):
        return self.block(x)

        
class dfcEncoderNet(nn.Module):

    def __init__(self, latent_size=100):
        super(dfcEncoderNet, self).__init__()
        self.latent_size = latent_size
        self.features = nn.Sequential(
            dfcEncoderBlock(3, 32),
            dfcEncoderBlock(32, 64),
            dfcEncoderBlock(64, 128),
            dfcEncoderBlock(128, 256)
        )
        self.logvar = nn.Linear(256 * 4 * 4, self.latent_size)
        self.mean = nn.Linear(256 * 4 * 4, self.latent_size)
        init_layers(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256 * 4 * 4)
        mean = self.mean(x)
        logvar = self.logvar(x) + magic_constant
        return mean, logvar


class dfcDecoderNet(nn.Module):

    def __init__(self, latent_size):
        super(dfcDecoderNet, self).__init__()
        self.dense = nn.Linear(latent_size, 256 * 4 * 4)
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            dfcDecoderBlock(256, 128),
            dfcDecoderBlock(128, 64),
            dfcDecoderBlock(64, 32),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(32, 3, kernel_size=(3,3),
                               padding=1, bias=False),
            nn.Sigmoid()
        )
        init_layers(self)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 256, 4, 4)
        x = self.relu(x)
        x = self.conv_block(x)
        return x


class dfcVAE(nn.Module):

    def __init__(self, latent_size=512):
        super(dfcVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = dfcEncoderNet(latent_size=latent_size)
        self.decoder = dfcDecoderNet(latent_size=latent_size)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        x = self.sample(mean, logvar)
        x = self.decoder(x)
        return x

    def sample(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.cuda.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mean)
    
    def generate(self, n_samples):
        mean = Variable(torch.zeros(n_samples, self.latent_size)).cuda()
        logvar = Variable(torch.zeros(n_samples, self.latent_size)).cuda()
        x = self.sample(mean, logvar)
        return self.decoder(x)

