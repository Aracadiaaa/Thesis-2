import os
import pandas as pd
import torch
import numpy as np

# ---------- PATHS (anchored to project root) ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # week4/Model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # week4

SCORES_CSV  = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")
ALIGNED_CSV = os.path.join(PROJECT_ROOT, "Data", "aligned_multimodal_rows.csv")  # must exist
ROBERTA_PT  = os.path.join(PROJECT_ROOT, "data", "processed", "roberta_embeddings.pt")  # adjust if needed
OUT_CSV     = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

# ---------- LOAD ----------
scores = pd.read_csv(SCORES_CSV)
df = pd.read_csv(ALIGNED_CSV)

# Ensure IDs are strings
scores["video_id"] = scores["video_id"].astype(str)
df["display_id"] = df["display_id"].astype(str)

# ---------- 1) MAX score per video ----------
max_scores = (
    scores.groupby("video_id")["recon_error"]
    .max()
    .reset_index()
    .rename(columns={"video_id": "display_id", "recon_error": "max_visual_score"})
)

# ---------- 2) Merge onto aligned rows ----------
merged = df.merge(max_scores, on="display_id", how="inner")
print("Rows after merge (should be ~4100):", len(merged))

# ---------- 3) Add RoBERTa vectors aligned by row_index ----------
text_emb = torch.load(ROBERTA_PT)  # [6000,768] or [4100,768]

if "row_index" not in merged.columns:
    raise ValueError("aligned_multimodal_rows.csv must contain 'row_index' column to align embeddings correctly.")

idx = merged["row_index"].astype(int).tolist()
text_keep = text_emb[idx].cpu().numpy()  # [N, 768]

# ---------- 4) Keep meta fields ----------
# Make sure view_count/like_count are numeric
for col in ["view_count", "like_count"]:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)

LABEL_COL = "relevance"  # change ONLY if your label column has a different name

if LABEL_COL not in merged.columns:
    raise ValueError(f"Label column '{LABEL_COL}' not found in aligned_multimodal_rows.csv")

meta_cols = [
    "display_id",
    "view_count",
    "like_count",
    "max_visual_score",
    LABEL_COL
]

meta = merged[meta_cols].reset_index(drop=True)

# ---------- 5) Flatten RoBERTa vectors into columns ----------
roberta_cols = [f"r{i}" for i in range(text_keep.shape[1])]
roberta_df = pd.DataFrame(text_keep, columns=roberta_cols)

final = pd.concat([meta, roberta_df], axis=1)

# ---------- SAVE ----------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
final.to_csv(OUT_CSV, index=False)

print("✅ Saved:", OUT_CSV)
print("Final shape:", final.shape)
