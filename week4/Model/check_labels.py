import os, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # week4/Model
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # week4
ALIGNED_CSV = os.path.join(PROJECT_ROOT, "Data", "aligned_multimodal_rows.csv")

df = pd.read_csv(ALIGNED_CSV)
print("Columns:\n", df.columns.tolist())
