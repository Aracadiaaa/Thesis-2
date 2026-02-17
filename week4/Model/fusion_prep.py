# week4/Model/fusion_prep.py

import os
import pandas as pd
import torch

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_CSV   = os.path.join(PROJECT_ROOT, "Data", "Final_Ready_Dataset.csv")
SCORES_CSV = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")
ROBERTA_PT = os.path.join(PROJECT_ROOT, "Data", "Processed", "roberta_embeddings.pt")
OUT_CSV    = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

# ================= COLUMN NAMES =================
ID_COL    = "display_id"
LABEL_COL = "label"
VIEWS_COL = "view_count"
LIKES_COL = "like_count"

# ================= LOAD FILES =================
df = pd.read_csv(DATA_CSV)
scores = pd.read_csv(SCORES_CSV)
text_emb = torch.load(ROBERTA_PT, map_location="cpu")

df[ID_COL] = df[ID_COL].astype(str)
scores["video_id"] = scores["video_id"].astype(str)

print("CSV rows:", len(df))
print("RoBERTa:", tuple(text_emb.shape))
print("Score rows:", len(scores))

# ================= VALIDATION =================
if text_emb.shape[0] != len(df):
    raise ValueError(
        f"RoBERTa rows ({text_emb.shape[0]}) != CSV rows ({len(df)})"
    )

# ================= AGGREGATE VISUAL SCORES =================
max_scores = (
    scores.groupby("video_id")["recon_error"]
    .max()
    .reset_index()
    .rename(columns={"video_id": ID_COL, "recon_error": "max_visual_score"})
)

# ================= MERGE =================
df_reset = df.reset_index().rename(columns={"index": "orig_idx"})
merged = df_reset.merge(max_scores, on=ID_COL, how="inner")

print("Rows with visual score:", len(merged))

# ================= CREATE META =================
meta = merged[[ID_COL, VIEWS_COL, LIKES_COL, "max_visual_score", LABEL_COL, "category", "orig_idx"]].copy()


# ================= FIX LABELS (STRING → INT) =================
# normalize label text
meta[LABEL_COL] = meta[LABEL_COL].astype(str).str.strip().str.lower()

label_map = {
    "1": 1, "begin": 1, "benign": 1, "safe": 1,
    "2": 2, "borderline": 2, "marginal": 2,
    "3": 3, "malicious": 3, "inappropriate": 3
}

meta[LABEL_COL] = meta[LABEL_COL].map(label_map)

before = len(meta)
meta = meta.dropna(subset=[LABEL_COL, "max_visual_score"]).copy()
print("Dropped invalid rows:", before - len(meta))

meta[LABEL_COL] = meta[LABEL_COL].astype(int)

# ================= ALIGN ROBERTA =================
orig_idx = meta["orig_idx"].astype(int).tolist()

text_keep = text_emb[orig_idx].numpy()

roberta_cols = [f"r{i}" for i in range(text_keep.shape[1])]
roberta_df = pd.DataFrame(text_keep, columns=roberta_cols)

# ================= FINAL =================
meta = meta.drop(columns=["orig_idx"]).reset_index(drop=True)
roberta_df = roberta_df.reset_index(drop=True)

final = pd.concat([meta, roberta_df], axis=1)

# ================= SAVE =================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
final.to_csv(OUT_CSV, index=False)

print("✅ Saved:", OUT_CSV)
print("Final shape:", final.shape)
