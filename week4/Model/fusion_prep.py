# week4/Model/fusion_prep.py
# Fuses:
# - RoBERTa embeddings (768) -> PCA reduced (100) to balance modalities
# - Visual VAE recon error features (per-frame + stats)
# - Metadata (views, likes, category)
# - Label (1/2/3)
#
# Output:
#   week4/Data/final_training_data.csv

import os
import pandas as pd
import torch
from sklearn.decomposition import PCA

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # week4/Model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # week4

DATA_CSV   = os.path.join(PROJECT_ROOT, "Data", "Final_Ready_Dataset.csv")
SCORES_CSV = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")
ROBERTA_PT = os.path.join(PROJECT_ROOT, "Data", "Processed", "roberta_embeddings.pt")
OUT_CSV    = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

# ================= COLUMN NAMES =================
ID_COL    = "display_id"
LABEL_COL = "label"
VIEWS_COL = "view_count"
LIKES_COL = "like_count"
CAT_COL   = "category"  # will auto-switch to "categories" if needed

# Visual frame types expected from filename parsing
VALID_FRAME_TYPES = {"thumb", "start", "mid", "end"}

# PCA config (reduce RoBERTa 768 -> 100)
PCA_COMPONENTS = 100

# ================= LOAD =================
df = pd.read_csv(DATA_CSV)
scores = pd.read_csv(SCORES_CSV)
text_emb = torch.load(ROBERTA_PT, map_location="cpu")

print("CSV rows:", len(df))
print("RoBERTa raw:", tuple(text_emb.shape))
print("Score rows:", len(scores))

# auto choose category column
if CAT_COL not in df.columns and "categories" in df.columns:
    CAT_COL = "categories"

# validations
for req in [ID_COL, LABEL_COL, VIEWS_COL, LIKES_COL, CAT_COL]:
    if req not in df.columns:
        raise ValueError(f"Missing column in Final_Ready_Dataset.csv: {req}")

if text_emb.shape[0] != len(df):
    raise ValueError(
        f"RoBERTa rows ({text_emb.shape[0]}) != CSV rows ({len(df)}). "
        f"Re-run extract_data_features.py on the same CSV."
    )

df[ID_COL] = df[ID_COL].astype(str)

# ================= TEXT PCA REDUCTION =================
text_np = text_emb.numpy()  # [N,768]
print("Original text shape:", text_np.shape)

pca = PCA(n_components=PCA_COMPONENTS, random_state=42)
text_reduced = pca.fit_transform(text_np)  # [N,100]

print("Reduced text shape:", text_reduced.shape)

# ================= VISUAL FEATURES =================
# Expected columns in scores: video_id, frame_type, recon_error
for req in ["video_id", "frame_type", "recon_error"]:
    if req not in scores.columns:
        raise ValueError(f"Missing column in vae_recon_scores.csv: {req}")

scores["video_id"] = scores["video_id"].astype(str)
scores["frame_type"] = scores["frame_type"].astype(str).str.lower().str.strip()

# normalize thumbnail naming inconsistency
scores["frame_type"] = scores["frame_type"].replace({
    "thumbnnails": "thumb",
    "thumbnail": "thumb",
    "thumbnails": "thumb"
})


# keep only expected frame types
scores = scores[scores["frame_type"].isin(VALID_FRAME_TYPES)].copy()

# per-frame mean score per video (handles duplicates)
pivot = (
    scores.pivot_table(
        index="video_id",
        columns="frame_type",
        values="recon_error",
        aggfunc="mean"
    )
    .reset_index()
    .rename(columns={"video_id": ID_COL})
)

# Ensure all expected frame columns exist
for col in ["thumb", "start", "mid", "end"]:
    if col not in pivot.columns:
        pivot[col] = pd.NA

frame_cols = ["thumb", "start", "mid", "end"]

pivot["vis_max"]  = pivot[frame_cols].max(axis=1)
pivot["vis_mean"] = pivot[frame_cols].mean(axis=1)
pivot["vis_std"]  = pivot[frame_cols].std(axis=1)

pivot = pivot.rename(columns={
    "thumb": "thumb_score",
    "start": "start_score",
    "mid": "mid_score",
    "end": "end_score",
})

VIS_COLS = ["thumb_score", "start_score", "mid_score", "end_score", "vis_max", "vis_mean", "vis_std"]

# ================= MERGE + KEEP ORIGINAL ORDER =================
df_reset = df.reset_index().rename(columns={"index": "orig_idx"})
merged = df_reset.merge(pivot, on=ID_COL, how="inner")

print("Rows with visual features:", len(merged))

# ================= BUILD META =================
meta_cols = [ID_COL, VIEWS_COL, LIKES_COL, CAT_COL, LABEL_COL, "orig_idx"] + VIS_COLS
for c in meta_cols:
    if c not in merged.columns:
        raise ValueError(f"Missing column after merge: {c}")

meta = merged[meta_cols].copy()

# numeric conversions
meta[VIEWS_COL] = pd.to_numeric(meta[VIEWS_COL], errors="coerce").fillna(0)
meta[LIKES_COL] = pd.to_numeric(meta[LIKES_COL], errors="coerce").fillna(0)
for c in VIS_COLS:
    meta[c] = pd.to_numeric(meta[c], errors="coerce")

# label mapping (numeric or string)
meta[LABEL_COL] = meta[LABEL_COL].astype(str).str.strip().str.lower()
label_map = {
    "1": 1, "begin": 1, "benign": 1, "safe": 1,
    "2": 2, "borderline": 2, "marginal": 2,
    "3": 3, "malicious": 3, "inappropriate": 3
}
meta[LABEL_COL] = meta[LABEL_COL].map(label_map)

# drop invalid rows
before = len(meta)
meta = meta.dropna(subset=[LABEL_COL, "vis_max"]).copy()
print("Dropped rows with missing labels/visual:", before - len(meta))
meta[LABEL_COL] = meta[LABEL_COL].astype(int)

# ================= ALIGN PCA TEXT USING orig_idx =================
orig_idx = meta["orig_idx"].astype(int).tolist()
text_keep = text_reduced[orig_idx]  # [N,100]

text_cols = [f"t{i}" for i in range(text_keep.shape[1])]
text_df = pd.DataFrame(text_keep, columns=text_cols)

# Final concat
meta = meta.drop(columns=["orig_idx"]).reset_index(drop=True)
text_df = text_df.reset_index(drop=True)

final = pd.concat([meta, text_df], axis=1)

# ================= SAVE =================
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
final.to_csv(OUT_CSV, index=False)

print("✅ Saved:", OUT_CSV)
print("Final shape:", final.shape)
print("First columns:", list(final.columns[:20]))
