import torch
import pandas as pd

# -------- Paths --------
csv_path = "Data/Master_Dataset.csv"
meta_path = "Data/video_image_metadata.csv"

roberta_path = "data/processed/roberta_embeddings.pt"   # 6000 x 768
vae_video_latents_path = "vae_latents_agg.pt"           # ~4100 x 128

# -------- Load --------
df = pd.read_csv(csv_path)
meta = pd.read_csv(meta_path)

text_emb = torch.load(roberta_path)        # [6000, 768]
img_emb = torch.load(vae_video_latents_path)  # [~4100, 128]

print("CSV rows:", len(df))
print("RoBERTa:", text_emb.shape)
print("VAE video latents:", img_emb.shape)

# -------- Build list of videos that have images --------
image_video_ids = meta["video_id"].dropna().unique().tolist()
image_video_ids_set = set(image_video_ids)

# -------- Filter CSV to only those videos --------
# Assumes your video id column is display_id
df["row_index"] = range(len(df))
df_keep = df[df["display_id"].astype(str).isin(image_video_ids_set)].copy()

print("Rows with both text+images:", len(df_keep))

# -------- Select matching RoBERTa rows --------
keep_indices = df_keep["row_index"].tolist()
text_keep = text_emb[keep_indices]  # [N, 768]

# -------- Now we must align image embeddings order to df_keep order --------
# For this to work, your img_emb must be saved in the SAME video_id order used during aggregation.
# If you also saved a video_id list when aggregating, use it here.
# If not, we’ll assume aggregation grouped in sorted video_id order (pandas groupby default is sorted).
video_ids_sorted = sorted(meta["video_id"].unique())
video_to_idx = {vid: i for i, vid in enumerate(video_ids_sorted)}

img_indices = [video_to_idx[str(v)] for v in df_keep["display_id"].astype(str).tolist()]
img_keep = img_emb[img_indices]  # [N, 128]

# -------- Final sanity check --------
assert text_keep.shape[0] == img_keep.shape[0], "Still mismatched after filtering!"
print("Final aligned shapes:", text_keep.shape, img_keep.shape)

# -------- Fuse --------
multimodal = torch.cat([text_keep, img_keep], dim=1)  # [N, 896]
print("Multimodal shape:", multimodal.shape)

torch.save(multimodal, "multimodal_embeddings_4100.pt")
df_keep.to_csv("Data/aligned_multimodal_rows.csv", index=False)

print("✅ Saved:")
print("- multimodal_embeddings_4100.pt")
print("- Data/aligned_multimodal_rows.csv")
