import os, pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_CSV = os.path.join(PROJECT_ROOT, "Data", "Final_Ready_Dataset.csv")

df = pd.read_csv(DATA_CSV)
print(df.columns.tolist())
