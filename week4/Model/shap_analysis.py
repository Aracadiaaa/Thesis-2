import os
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")

# ================= LOAD =================
df = pd.read_csv(DATA)

# encode all non-numeric cols (except id)
for col in df.select_dtypes(include=["object", "string"]).columns:
    if col != "display_id":
        df[col] = df[col].astype("category").cat.codes

LABEL_COL = "label"
y = df[LABEL_COL].astype(int)
X = df.drop(columns=[LABEL_COL, "display_id"], errors="ignore")

# train same as best model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=500,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# ================= SHAP =================
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# ---- robust handling of SHAP output formats ----
# Possible formats:
# 1) list of arrays: [ (n_samples,n_features), ... per class ]
# 2) single array: (n_samples,n_features,n_classes)
# 3) Explanation object in newer SHAP

malicious_class = 3
classes = list(clf.classes_)
mal_idx = classes.index(malicious_class)

if isinstance(shap_values, list):
    sv = shap_values[mal_idx]  # (n_samples, n_features)
else:
    sv = shap_values
    if sv.ndim == 3:
        sv = sv[:, :, mal_idx]  # (n_samples, n_features)

# final sanity check
sv = np.array(sv)
assert sv.shape[0] == X_test.shape[0], f"Sample mismatch: {sv.shape} vs {X_test.shape}"
assert sv.shape[1] == X_test.shape[1], f"Feature mismatch: {sv.shape} vs {X_test.shape}"

# plot
plt.figure()
shap.summary_plot(sv, X_test, show=False, max_display=20)
out_path = os.path.join(BASE_DIR, "shap_summary_malicious.png")
plt.tight_layout()
plt.savefig(out_path, dpi=300)
print("✅ Saved:", out_path)

# also print top 10 features by mean |SHAP|
mean_abs = np.abs(sv).mean(axis=0)
top_idx = np.argsort(mean_abs)[::-1][:10]
print("\nTop 10 features (mean |SHAP|):")
for i in top_idx:
    print(f"{X_test.columns[i]}: {mean_abs[i]:.6f}")

print("\nVisual feature importance:")
if "max_visual_score" in X_test.columns:
    idx = list(X_test.columns).index("max_visual_score")
    print("max_visual_score:", mean_abs[idx])

# ================= VISUAL FEATURE RANKING =================
visual_feats = ["vis_max","vis_mean","vis_std",
                "thumb_score","start_score","mid_score","end_score"]

imp = mean_abs
names = list(X_test.columns)

# rank all features
ranked = sorted(
    [(names[i], imp[i]) for i in range(len(names))],
    key=lambda x: x[1],
    reverse=True
)

print("\nVisual Feature Rankings:")
for vf in visual_feats:
    for r, (name, val) in enumerate(ranked, start=1):
        if name == vf:
            print(f"{vf} -> rank #{r}, mean|SHAP| = {val:.6f}")
            break

