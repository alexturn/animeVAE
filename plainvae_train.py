import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from datasets import CelebDataset
from plainVAE import plainVAE

from tensorboardX import SummaryWriter
from tqdm import tqdm


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
torch.sum

alpha = .5
def loss_function(x, recon_x, mu, logvar):
    REC = torch.mean(torch.sum(((recon_x - x) ** 2).view(x.size(0), -1), dim=1))
    KLD = torch.mean(torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1), dim=1))
    return REC + alpha * KLD, REC, KLD


data = DataLoader(CelebDataset(path='img_align_celeba/'), batch_size=64, shuffle=True)
model = plainVAE(latent_size=512).cuda()
optimizer = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.999))
writer = SummaryWriter(log_dir='./PlainVAE')

n_epoch = 30
model.train()
for epoch in range(1, n_epoch + 1):
    print('Epoch number {} .........'.format(epoch))
    step = 0
    for x in tqdm(data):
        x = x.float().cuda()
        mu, logvar = model.encoder(x)
        latent_z = model.sample(mu, logvar)
        recon_x = model.decoder(latent_z)
        loss, rec, kld = loss_function(x, recon_x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        writer.add_scalar('ELBO', loss, (epoch - 1) * data.__len__() * 64 + step * 64)
        writer.add_scalar('REC', rec, (epoch - 1) * data.__len__() * 64 + step * 64)
        writer.add_scalar('KLD', kld, (epoch - 1) * data.__len__() * 64 + step * 64)

        if step % 300 == 0:
            model.eval()
            samples = model.generate(64)
            writer.add_image('Generated images', make_grid(samples), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            writer.add_image('Input image', make_grid(x), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            writer.add_image('Reconstructed image', make_grid(recon_x), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            model.train()

torch.save(model, './PlainVAE/dump.model')

