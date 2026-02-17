import os
import sys
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from vae_model import ConvVAE  # your VAE class
from torch.utils.data import Dataset
from PIL import Image
import glob

# ================= PATH SETUP =================
IMG_ROOT = r"C:\Users\JC\Desktop\Thesis\week4\Data\Images"
MODEL_PATH = r"week4\vae_trained.pth"
OUT_CSV = r"week4/Data/vae_recon_scores.csv"

print("Using IMG_ROOT:", IMG_ROOT)
print("Exists?", os.path.exists(IMG_ROOT))
# ============================================

# -------- settings --------
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # assumes images already 224x224
])
class FlatImageDataset(Dataset):
    def __init__(self, root, transform=None):
        self.paths = glob.glob(os.path.join(root, "*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path


dataset = FlatImageDataset(IMG_ROOT, transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# Get file paths in exact order as ImageFolder outputs
all_paths = dataset.paths

model = ConvVAE().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

rows = []
with torch.no_grad():
    idx = 0
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon, mu, logvar = model(imgs)

        # per-image reconstruction error (MSE over pixels)
        # reduction none -> mean over C,H,W
        per_img = F.mse_loss(recon, imgs, reduction="none")
        per_img = per_img.mean(dim=(1,2,3)).detach().cpu().numpy()

        for s in per_img:
            img_path = all_paths[idx]
            img_name = os.path.basename(img_path)
            base = os.path.splitext(img_name)[0]

            # parse video_id + frame_type from videoID_start.jpg
            if "_" in base:
                video_id, frame_type = base.rsplit("_", 1)
            else:
                video_id, frame_type = base, "unknown"

            rows.append({
                "image_path": img_path,
                "image_name": img_name,
                "video_id": video_id,
                "frame_type": frame_type,
                "recon_error": float(s),
            })
            idx += 1

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print("✅ Saved:", OUT_CSV)
print(df.head())
print("Total images scored:", len(df))
print("Using IMG_ROOT:", IMG_ROOT)
print("Exists?", os.path.exists(IMG_ROOT))