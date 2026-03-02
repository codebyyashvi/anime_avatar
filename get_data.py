import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

DATASET_NAME = "murali1729S/portrait_2_avatar"
OUTPUT_DIR = "dataset"
IMG_SIZE = (256, 256)
VAL_SPLIT = 0.1
SEED = 42

train_real = os.path.join(OUTPUT_DIR, "train", "real")
train_anime = os.path.join(OUTPUT_DIR, "train", "anime")
val_real = os.path.join(OUTPUT_DIR, "val", "real")
val_anime = os.path.join(OUTPUT_DIR, "val", "anime")

for path in [train_real, train_anime, val_real, val_anime]:
    os.makedirs(path, exist_ok=True)

print("🔹 Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

print("🔹 Creating train/validation split...")
split = dataset.train_test_split(
    test_size=VAL_SPLIT,
    seed=SEED
)

train_ds = split["train"]
val_ds = split["test"]

def save_dataset(ds, real_dir, anime_dir):
    for idx, item in enumerate(tqdm(ds)):
        real_img = item["input_image"]
        anime_img = item["edited_image"]

        # Ensure RGB
        real_img = real_img.convert("RGB").resize(IMG_SIZE)
        anime_img = anime_img.convert("RGB").resize(IMG_SIZE)

        real_img.save(os.path.join(real_dir, f"{idx}.png"))
        anime_img.save(os.path.join(anime_dir, f"{idx}.png"))

print("🔹 Saving training images...")
save_dataset(train_ds, train_real, train_anime)

print("🔹 Saving validation images...")
save_dataset(val_ds, val_real, val_anime)

print("\n✅ Dataset successfully saved in 'dataset/' folder")