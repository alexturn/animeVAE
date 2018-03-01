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


class plainEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(plainEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5,5), 
                      stride=(2,2), padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        init_layers(self)

    def forward(self, x):
        return self.block(x)


class plainDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(plainDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(5,5),
                               stride=(2,2), padding=2, output_padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        init_layers(self)

    def forward(self, x):
        return self.block(x)

        
class plainEncoderNet(nn.Module):

    def __init__(self, latent_size=100):
        super(plainEncoderNet, self).__init__()
        self.latent_size = latent_size
        self.mean_bn = nn.BatchNorm1d(self.latent_size)
        self.logvar_bn = nn.BatchNorm1d(self.latent_size)
        self.softplus = nn.Softplus()
        self.features = nn.Sequential(
            plainEncoderBlock(3, 64),
            plainEncoderBlock(64, 128),
            plainEncoderBlock(128, 256),
            plainEncoderBlock(256, 512)
        )
        self.logvar = nn.Linear(512 * 4 * 4, self.latent_size)
        self.mean = nn.Linear(512 * 4 * 4, self.latent_size)
        init_layers(self)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        mean = self.mean_bn(self.mean(x))
        logvar = self.softplus(self.logvar_bn(self.logvar(x))) + magic_constant
        return mean, logvar


class plainDecoderNet(nn.Module):

    def __init__(self, latent_size):
        super(plainDecoderNet, self).__init__()
        self.dense = nn.Linear(latent_size, 512 * 4 * 4)
        self.bn = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        self.conv_block = nn.Sequential(
            plainDecoderBlock(512, 256),
            plainDecoderBlock(256, 128),
            plainDecoderBlock(128, 64),
            nn.ConvTranspose2d(64, 3, kernel_size=(4,4),
                               stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        init_layers(self)

    def forward(self, x):
        x = self.dense(x)
        x = x.view(-1, 512, 4, 4)
        x = self.relu(self.bn(x))
        x = self.conv_block(x)
        return x


class plainVAE(nn.Module):

    def __init__(self, latent_size=512):
        super(plainVAE, self).__init__()
        self.latent_size = latent_size
        self.encoder = plainEncoderNet(latent_size=latent_size)
        self.decoder = plainDecoderNet(latent_size=latent_size)

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

