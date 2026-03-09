import os
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rf_model.pkl")
DATA_PATH = os.path.join(PROJECT_ROOT, "Data", "original_test_unlabeled.csv")
OUT_PATH = os.path.join(PROJECT_ROOT, "Data", "original_test_predictions.csv")

bundle = joblib.load(MODEL_PATH)
clf = bundle["model"]
feature_cols = bundle["feature_cols"]

df = pd.read_csv(DATA_PATH)

# Keep display_id for UI
display_ids = df["display_id"].copy() if "display_id" in df.columns else None

# Drop display_id before prediction
X = df.drop(columns=["display_id"], errors="ignore").copy()

# Encode all string columns the same way training did
for col in X.select_dtypes(include=["object", "string"]).columns:
    X[col] = X[col].astype("category").cat.codes

# Align columns to training feature space
X = X.reindex(columns=feature_cols, fill_value=0)

# Predict
probs = clf.predict_proba(X)
preds = clf.predict(X)

out_df = df.copy()
out_df["predicted_label"] = preds
out_df["prob_benign"] = probs[:, 0]
out_df["prob_borderline"] = probs[:, 1]
out_df["prob_malicious"] = probs[:, 2]

out_df.to_csv(OUT_PATH, index=False)
print("✅ Saved:", OUT_PATH)
print(out_df.head())