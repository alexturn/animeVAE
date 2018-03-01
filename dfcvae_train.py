import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid

from datasets import CelebDataset
from dfcVAE import dfcVAE
from VGG import VGG


from tensorboardX import SummaryWriter
from tqdm import tqdm


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

alpha = .01
descriptor = VGG().cuda()
def loss_function(x, recon_x, mu, logvar):
    KLD = torch.mean(torch.sum(0.5 * (mu ** 2 + torch.exp(logvar) - logvar - 1), dim=1))

    REC = 0.
    output_recon, output_target = descriptor(recon_x), descriptor(x)
    for i in range(len(output_recon)):
        REC += torch.mean((output_recon[i] - output_target[i].detach()) ** 2)

    return REC + alpha * KLD, REC, KLD


data = DataLoader(CelebDataset(path='img_align_celeba/'), batch_size=64, shuffle=True)
model = dfcVAE(latent_size=100).cuda()
optimizer = optim.Adam(model.parameters(), lr=5e-4, betas=(0.5, 0.999))
writer = SummaryWriter(log_dir='./dfcVAE_10')

n_epoch = 10
model.train()
for epoch in range(1, n_epoch + 1):
    print('Epoch number {} .........'.format(epoch))
    step = 0
    for x in tqdm(data):
        x = x.float().cuda()
        mu, logvar = model.encoder(x)
        latent_z = model.sample(mu, logvar)
        recon_x = model.decoder(latent_z)
        # print(x.min(), x.max())
        # print(recon_x.min(), recon_x.max())
        loss, rec, kld = loss_function(x, recon_x, mu, logvar)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        writer.add_scalar('ELBO', loss, (epoch - 1) * data.__len__() * 64 + step * 64)
        writer.add_scalar('REC', rec, (epoch - 1) * data.__len__() * 64 + step * 64)
        writer.add_scalar('KLD', kld, (epoch - 1) * data.__len__() * 64 + step * 64)

        if step % 1000 == 0:
            model.eval()
            samples = model.generate(64)
            writer.add_image('Generated images', make_grid(samples), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            writer.add_image('Input image', make_grid(x), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            writer.add_image('Reconstructed image', make_grid(recon_x), 
                             (epoch - 1) * data.__len__() * 64 + step * 64)
            model.train()

torch.save(model, './dfcVAE/dump.model')

