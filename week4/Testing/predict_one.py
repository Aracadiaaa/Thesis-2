import os
import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_CSV  = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")
MODEL_PKL = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")

LABEL_COL = "label"
ID_COL = "display_id"

# ---------------- Load data ----------------
df = pd.read_csv(DATA_CSV)

# ---------------- Load trained model bundle ----------------
# We will save/load a bundle that includes:
# - model
# - feature_columns (after get_dummies)
bundle = joblib.load(MODEL_PKL)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]

# ---------------- Choose ONE row ----------------
# Option A: pick by index
row_idx = 0  # change this
row = df.iloc[[row_idx]].copy()

# Option B: pick by display_id
# target_id = "B4CTq_-LRfU"
# row = df[df[ID_COL] == target_id].head(1).copy()

if row.empty:
    raise ValueError("Row not found. Check row index or display_id.")

# ---------------- Prepare features exactly like training ----------------
y_true = row[LABEL_COL].values[0] if LABEL_COL in row.columns else None

X_one = row.drop(columns=[LABEL_COL], errors="ignore")
video_id = X_one[ID_COL].values[0] if ID_COL in X_one.columns else None
X_one = X_one.drop(columns=[ID_COL], errors="ignore")

# one-hot encode categoricals (like category)
X_one = pd.get_dummies(X_one, columns=X_one.select_dtypes(include=["object"]).columns, drop_first=False)

# align columns to training feature space
X_one = X_one.reindex(columns=feature_cols, fill_value=0)

# ---------------- Predict ----------------
pred = clf.predict(X_one)[0]
proba = clf.predict_proba(X_one)[0]
classes = clf.classes_

print("==== Single Row Prediction ====")
print("display_id:", video_id)
print("Predicted class:", pred)
print("Probabilities:")
for c, p in zip(classes, proba):
    print(f"  class {c}: {p:.4f}")

if y_true is not None:
    print("True label:", y_true)

# Optional: map class numbers to names
name_map = {1: "Benign", 2: "Borderline", 3: "Malicious"}
print("Predicted meaning:", name_map.get(int(pred), str(pred)))