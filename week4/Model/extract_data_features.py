import os
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# ===== PATHS =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # week4/Model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # week4

CSV_PATH = os.path.join(PROJECT_ROOT, "Data", "Final_Ready_Dataset.csv")  # <-- change if new csv name
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PT  = os.path.join(OUT_DIR, "roberta_embeddings.pt")
OUT_NPY = os.path.join(OUT_DIR, "roberta_embeddings.npy")
OUT_IDS = os.path.join(OUT_DIR, "roberta_video_ids.csv")

# ===== SETTINGS =====
ID_COL = "display_id"       # <-- must exist in your CSV
MAX_LEN = 128
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base").to(device)
model.eval()

# ===== LOAD DATA =====
df = pd.read_csv(CSV_PATH)
print("CSV rows:", len(df))

# Make sure ID exists
if ID_COL not in df.columns:
    raise ValueError(f"Missing ID column '{ID_COL}' in CSV")

df[ID_COL] = df[ID_COL].astype(str)

# Build text input
texts = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()

# ===== EMBEDDING EXTRACTION (BATCHED) =====
all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting RoBERTa"):
    batch_texts = texts[i:i+BATCH_SIZE]
    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token [batch,768]

    all_embeddings.append(emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0)
print("Final embedding shape:", embeddings.shape)

# ===== SAVE =====
torch.save(embeddings, r"week4/Data/Processed/roberta_embeddings.pt")

df[["display_id"]].to_csv(
    r"week4/Data/Processed/roberta_video_ids.csv",
    index=False
)

print("✅ Saved:")
print("-", OUT_PT)
print("-", OUT_NPY)
print("-", OUT_IDS)
