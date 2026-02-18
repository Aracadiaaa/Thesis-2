import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vae_model import ConvVAE, vae_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Your actual folder:
# Data/Classified data/Begin/  (contains images)
TRAIN_ROOT = os.path.join(PROJECT_ROOT, "Data", "Classified_Dataset", "Benign")

OUT_MODEL  = os.path.join(PROJECT_ROOT, "models", "vae_v1.pth")

EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists(TRAIN_ROOT):
    raise FileNotFoundError(f"TRAIN_ROOT not found: {TRAIN_ROOT}")

# IMPORTANT: ImageFolder needs subfolders, but since TRAIN_ROOT is a single class folder,
# we wrap it by using its parent and selecting only Begin. Easiest way:
PARENT = os.path.dirname(TRAIN_ROOT)  # Data/Classified data
dataset_all = datasets.ImageFolder(root=PARENT, transform=transform)

# filter only Begin samples
begin_class_idx = dataset_all.class_to_idx.get("Benign")

if begin_class_idx is None:
    raise ValueError(f"'Benign' folder not detected in {PARENT}. Found: {dataset_all.classes}")

begin_samples = [(p, y) for (p, y) in dataset_all.samples if y == begin_class_idx]
dataset_all.samples = begin_samples
dataset_all.targets = [y for (_, y) in begin_samples]

loader = DataLoader(dataset_all, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

print("Train root (Begin only):", TRAIN_ROOT)
print("Begin images:", len(dataset_all))
print("Device:", DEVICE)

model = ConvVAE().to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LR)

model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon, mu, logvar = model(imgs)
        loss = vae_loss(recon, imgs, mu, logvar)

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()

    print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {total/len(loader):.6f}")

os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
torch.save(model.state_dict(), OUT_MODEL)
print("✅ Saved:", OUT_MODEL)
