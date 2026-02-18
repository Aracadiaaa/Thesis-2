import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from vae_model import ConvVAE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

IMG_ROOT   = os.path.join(PROJECT_ROOT, "Data", "Classified_Dataset")   # contains Begin/Borderline/Malicious
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "vae_v1.pth")
OUT_CSV    = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.ImageFolder(root=IMG_ROOT, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

all_paths = [p for (p, _) in dataset.samples]

model = ConvVAE().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

def normalize_frame(ft: str) -> str:
    ft = ft.lower().strip()
    if ft in ["thumbnnails", "thumbnail", "thumbnails", "thumb"]:
        return "thumb"
    return ft

rows = []
idx = 0
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)
        recon, mu, logvar = model(imgs)

        per_img = F.mse_loss(recon, imgs, reduction="none").mean(dim=(1,2,3)).detach().cpu().numpy()

        for s in per_img:
            img_path = all_paths[idx]
            img_name = os.path.basename(img_path)
            base = os.path.splitext(img_name)[0]

            # parse: videoID_start / videoID_mid / videoID_end / videoID_thumbnnails
            if "_" in base:
                video_id, frame_type = base.rsplit("_", 1)
            else:
                video_id, frame_type = base, "unknown"

            frame_type = normalize_frame(frame_type)

            rows.append({
                "image_path": img_path,
                "image_name": img_name,
                "video_id": video_id,
                "frame_type": frame_type,
                "recon_error": float(s),
            })
            idx += 1

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
df.to_csv(OUT_CSV, index=False)

print("✅ Saved:", OUT_CSV)
print("Total images scored:", len(df))
print(df["frame_type"].value_counts().head(10))
print("Classes found:", dataset.classes)
