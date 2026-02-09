import os
import pandas as pd

BASE_DIR = "Data/dataset_images"

rows = []

for category in os.listdir(BASE_DIR):
    category_path = os.path.join(BASE_DIR, category)

    if not os.path.isdir(category_path):
        continue

    for img_name in os.listdir(category_path):
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        # Remove extension
        base = os.path.splitext(img_name)[0]

        # Expect: videoID_start / videoID_mid / videoID_end
        parts = base.rsplit("_", 1)
        if len(parts) != 2:
            continue

        video_id, frame_type = parts

        rows.append({
            "video_id": video_id,
            "frame_type": frame_type,
            "category": category,
            "image_name": img_name,
            "image_path": os.path.join(category_path, img_name)
        })

df = pd.DataFrame(rows)

df.to_csv("Data/video_image_metadata.csv", index=False)

print("✅ Metadata created")
print("Columns:", df.columns.tolist())
print("Total images:", len(df))
print("Total videos:", df["video_id"].nunique())
