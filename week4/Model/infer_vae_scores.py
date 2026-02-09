import os
import sys
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from vae_model import ConvVAE  # your VAE class

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # week4/Model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))  # week4
IMG_ROOT = os.path.join(PROJECT_ROOT, "Data", "dataset_images")
print("Using IMG_ROOT:", IMG_ROOT)
print("Exists?", os.path.exists(IMG_ROOT))
# ============================================

# -------- paths --------
MODEL_PATH = os.path.join(PROJECT_ROOT, "vae_trained.pth")  # change if needed
OUT_CSV    = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")      # category subfolders
OUT_CSV = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")


# -------- settings --------
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),  # assumes images already 224x224
])

dataset = datasets.ImageFolder(root=IMG_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Get file paths in exact order as ImageFolder outputs
all_paths = [p for (p, _) in dataset.samples]

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