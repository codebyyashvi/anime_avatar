import torch
from torch.utils.data import DataLoader
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
from utils import AnimeDataset
import torch.nn as nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

G = UNetGenerator().to(device)
D = PatchDiscriminator().to(device)

criterion_gan = nn.MSELoss()
criterion_l1 = nn.L1Loss()

opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

dataset = AnimeDataset(
    "dataset/train/real",
    "dataset/train/anime"
)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

for epoch in range(50):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}")
    for real, anime in loop:
        real, anime = real.to(device), anime.to(device)

        fake = G(real)
        real_pred = D(real, anime)
        fake_pred = D(real, fake.detach())

        loss_D = (
            criterion_gan(real_pred, torch.ones_like(real_pred)) +
            criterion_gan(fake_pred, torch.zeros_like(fake_pred))
        ) * 0.5

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        fake_pred = D(real, fake)
        loss_G = (
            criterion_gan(fake_pred, torch.ones_like(fake_pred)) +
            100 * criterion_l1(fake, anime)
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loop.set_postfix(G=loss_G.item(), D=loss_D.item())

    torch.save(G.state_dict(), f"generator_epoch_{epoch+1}.pth")