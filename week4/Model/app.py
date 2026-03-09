import os
import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="YouTube Video Prediction Viewer", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "original_test_predictions.csv")
TEST_IMG_ROOT = os.path.join(PROJECT_ROOT, "Data", "Images")

LABEL_MAP = {
    1: "Benign",
    2: "Borderline",
    3: "Malicious"
}

@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)

df = load_data()

st.title("YouTube Video Prediction Viewer")
st.write(f"Total rows: {len(df)}")

row_idx = st.number_input("Select row index", min_value=0, max_value=len(df)-1, value=0, step=1)
row = df.iloc[int(row_idx)]

# ================= BIG PREDICTED LABEL =================
pred_label = row.get("predicted_label", None)

if pd.notna(pred_label):
    pred_label = int(pred_label)
    pred_text = LABEL_MAP.get(pred_label, str(pred_label))

    st.markdown(
        f"""
        <div style="text-align:center; padding:20px; border-radius:12px; background-color:#000000;">
            <h1 style="font-size:42px; margin-bottom:0;">{pred_text}</h1>
            <p style="font-size:18px; color:gray; margin-top:5px;">Predicted Classification</p>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.error("No predicted_label column found in CSV.")

# ================= SMALLER PROBABILITIES =================
st.write("### Prediction Probabilities")

prob_map = {}
if "prob_benign" in row.index:
    prob_map["Benign"] = float(row["prob_benign"])
if "prob_borderline" in row.index:
    prob_map["Borderline"] = float(row["prob_borderline"])
if "prob_malicious" in row.index:
    prob_map["Malicious"] = float(row["prob_malicious"])

st.write(prob_map)

# ================= METADATA =================
for col in ["display_id", "title", "description", "category", "categories"]:
    if col in row.index and pd.notna(row[col]):
        st.write(f"**{col}:** {row[col]}")

# ================= IMAGE LOOKUP =================
def find_image(video_id, suffix):
    # normalize naming
    if suffix == "thumb":
        suffix = "thumbnail"

    extensions = ["jpg", "jpeg", "png", "webp"]

    for ext in extensions:
        path = os.path.join(TEST_IMG_ROOT, f"{video_id}_{suffix}.{ext}")
        if os.path.exists(path):
            return path

    return None

# ================= SHOW 4 IMAGES =================
video_id = row.get("display_id", None)

st.write("### Video Frames")
cols = st.columns(4)

frame_info = [
    ("Thumbnail", "thumb"),
    ("Start", "start"),
    ("Middle", "mid"),
    ("End", "end"),
]

for i, (label, suffix) in enumerate(frame_info):
    with cols[i]:
        if pd.notna(video_id):
            img_path = find_image(video_id, suffix)
            if img_path is not None:
                img = Image.open(img_path)
                st.image(img, caption=label, use_container_width=True)
            else:
                st.warning(f"{label} image not found")
        else:
            st.warning("No display_id found")