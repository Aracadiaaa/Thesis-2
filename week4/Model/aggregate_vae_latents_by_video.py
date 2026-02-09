import torch
import pandas as pd
import os

# ========= CONFIG =========
vae_latents_path = "vae_latents.pt"          # all 12,300 latent vectors
metadata_csv = "Data/video_image_metadata.csv"  # CSV with 'video_id' and 'image_file'
output_path = "vae_latents_agg.pt"          # aggregated per video
pooling = "mean"                             # 'mean' or 'max'

# ========= LOAD =========
vae_latents = torch.load(vae_latents_path)  # [12300, 128]
metadata = pd.read_csv(metadata_csv)

# Make sure ordering matches
assert len(metadata) == vae_latents.shape[0], "Metadata and VAE latents length mismatch!"

# ========= AGGREGATE =========
agg_latents = []
agg_video_ids = []

for vid, group in metadata.groupby("video_id"):
    indices = group.index.tolist()
    z = vae_latents[indices]  # [3, 128] if 3 frames per video
    if pooling == "mean":
        z_video = z.mean(dim=0)
    elif pooling == "max":
        z_video, _ = z.max(dim=0)
    else:
        raise ValueError("Pooling must be 'mean' or 'max'")
    
    agg_latents.append(z_video)
    agg_video_ids.append(vid)

agg_latents = torch.stack(agg_latents, dim=0)  # [num_videos, 128]

print("Aggregated VAE latents:", agg_latents.shape)
print("Number of unique videos:", len(agg_video_ids))

# ========= SAVE =========
torch.save(agg_latents, output_path)
print(f"✅ Saved aggregated latents to {output_path}")
