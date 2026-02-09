import torch
from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import numpy as np
from tqdm import tqdm

# ================== DEVICE ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== LOAD MODEL ==================
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaModel.from_pretrained("roberta-base")
model.to(device)
model.eval()

# ================== EMBEDDING FUNCTION ==================
def extract_embeddings_batched(texts, batch_size=16, max_len=128):
    """
    texts: list of strings
    returns: torch.Tensor of shape (N, 768)
    """
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting RoBERTa embeddings"):
        batch_texts = texts[i:i + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            # CLS token embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(batch_embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)

# ================== LOAD DATA ==================
df = pd.read_csv("Data/Master_Dataset.csv")
print("Total rows in CSV:", len(df))

texts = (
    df["title"].fillna("") + " " +
    df["description"].fillna("")
).tolist()

# ================== EXTRACT ==================
embeddings = extract_embeddings_batched(
    texts,
    batch_size=16,   # safe default (use 32 if GPU has enough VRAM)
    max_len=128
)

# ================== SAVE ==================
print("Final embedding shape:", embeddings.shape)

np.save("data/processed/roberta_embeddings.npy", embeddings.numpy())
torch.save(embeddings, "data/processed/roberta_embeddings.pt")

print("✅ Saved:")
print("- data/processed/roberta_embeddings.npy")
print("- data/processed/roberta_embeddings.pt")
