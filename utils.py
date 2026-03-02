import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class AnimeDataset(Dataset):
    def __init__(self, real_dir, anime_dir):
        self.real_dir = real_dir
        self.anime_dir = anime_dir
        self.files = sorted(os.listdir(real_dir))
        self.transform = T.Compose([
            T.Resize((256,256)),
            T.ToTensor(),
            T.Normalize((0.5,)*3, (0.5,)*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        real = Image.open(os.path.join(self.real_dir, self.files[idx])).convert("RGB")
        anime = Image.open(os.path.join(self.anime_dir, self.files[idx])).convert("RGB")
        return self.transform(real), self.transform(anime)