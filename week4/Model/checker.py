import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

df = pd.read_csv(r"week4\Data\aligned_multimodal_rows.csv")
print(df.columns)
print(df["display_id"].head())

# --- paths ---
CSV_PATH = r"week4\Data\aligned_multimodal_rows.csv"   # or your labeled/aligned csv
IMG_ROOT = r"week4\Data\Images"               # parent folder containing category subfolders

VID_COL = "display_id"  # video id column in CSV
NEEDED = ["thumb", "start", "mid", "end"]

# --- load csv ---
df = pd.read_csv(CSV_PATH)
df[VID_COL] = df[VID_COL].astype(str)

# --- build lookup: video_id -> {frame: path} ---
img_map = {}
valid_ext = (".jpg", ".jpeg", ".png", ".webp")

def norm(ft):
    ft = ft.lower().strip()
    if ft in ["thumbnail", "thumb", "tn"]:
        return "thumb"
    return ft

for cat in os.listdir(IMG_ROOT):
    cat_path = os.path.join(IMG_ROOT, cat)
    if not os.path.isdir(cat_path):
        continue
    for fn in os.listdir(cat_path):
        if not fn.lower().endswith(valid_ext):
            continue
        base = os.path.splitext(fn)[0]
        if "_" not in base:
            continue
        vid, ft = base.rsplit("_", 1)
        ft = norm(ft)
        img_map.setdefault(vid, {})
        # keep first seen per frame type
        img_map[vid].setdefault(ft, os.path.join(cat_path, fn))

# --- filter to rows that have at least 1 image ---
df_ok = df[df[VID_COL].isin(img_map.keys())].copy()
print("CSV rows:", len(df))
print("Rows with any images found:", len(df_ok))

# --- sample some to view ---
sample = df_ok.sample(n=min(5, len(df_ok)), random_state=42)

for _, row in sample.iterrows():
    vid = str(row[VID_COL])
    title = str(row.get("title", ""))
    desc = str(row.get("description", ""))[:160]

    frames = img_map.get(vid, {})
    print("\n==============================")
    print("VIDEO ID:", vid)
    print("TITLE:", title)
    print("DESC:", desc)
    print("Frames found:", sorted(frames.keys()))

    plt.figure(figsize=(14, 4))
    for i, ft in enumerate(NEEDED, start=1):
        plt.subplot(1, 4, i)
        p = frames.get(ft)
        if p and os.path.exists(p):
            img = Image.open(p).convert("RGB")
            plt.imshow(img)
            plt.title(ft)
        else:
            plt.text(0.5, 0.5, f"Missing {ft}", ha="center", va="center")
            plt.title(ft)
        plt.axis("off")
    plt.suptitle(f"{vid} | {title}", y=1.05)
    plt.tight_layout()
    plt.show()
