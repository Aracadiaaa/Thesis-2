import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_CSV = os.path.join(PROJECT_ROOT, "Data", "final_training_data.csv")
SCORES_CSV = os.path.join(PROJECT_ROOT, "Data", "vae_recon_scores.csv")

IMG_ROOT_1 = os.path.join(PROJECT_ROOT, "Data", "Classified data")         # if you use this
IMG_ROOT_2 = os.path.join(PROJECT_ROOT, "Data", "Classified_Dataset")      # if you use this

OUT_DIR = os.path.join(PROJECT_ROOT, "Results")
os.makedirs(OUT_DIR, exist_ok=True)

# ================= LOAD =================
df = pd.read_csv(DATA_CSV)

LABEL_COL = "label"
ID_COL = "display_id"

if LABEL_COL not in df.columns:
    raise ValueError("label column not found in final_training_data.csv")

y = df[LABEL_COL].astype(int)
X = df.drop(columns=[LABEL_COL, ID_COL], errors="ignore")

# one-hot encode any non-numeric columns (e.g., category)
X = pd.get_dummies(X, columns=X.select_dtypes(include=["object"]).columns, drop_first=False)

# make sure everything is numeric
X = X.apply(pd.to_numeric, errors="coerce").fillna(0)


# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= TRAIN RF =================
clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# ================= 1) CONFUSION MATRIX =================
cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))

plt.figure(figsize=(6, 5))
disp.plot(values_format="d")
plt.title("Confusion Matrix (Random Forest)")
plt.tight_layout()
cm_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(cm_path, dpi=300)
plt.close()
print("✅ Saved:", cm_path)

# ================= 2) ROC CURVE (Multiclass OVR) =================
# Binarize labels for multiclass ROC
classes = sorted(y.unique())
y_test_bin = label_binarize(y_test, classes=classes)

# predict probabilities
y_proba = clf.predict_proba(X_test)  # list for multiclass
y_proba = np.array(y_proba).T if isinstance(y_proba, list) else y_proba

plt.figure(figsize=(7, 6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {c} (AUC={roc_auc:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest)")
plt.legend(loc="lower right")
plt.tight_layout()

roc_path = os.path.join(OUT_DIR, "roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.close()
print("✅ Saved:", roc_path)

# ================= 3) EXAMPLES GRID (Most anomalous vs most normal) =================
# Uses vae_recon_scores.csv -> image_path + recon_error

if os.path.exists(SCORES_CSV):
    s = pd.read_csv(SCORES_CSV)

    # Find the most anomalous and most normal images globally
    s = s.dropna(subset=["image_path", "recon_error"]).copy()
    s["recon_error"] = pd.to_numeric(s["recon_error"], errors="coerce")
    s = s.dropna(subset=["recon_error"]).copy()

    top_anom = s.sort_values("recon_error", ascending=False).head(9)
    top_norm = s.sort_values("recon_error", ascending=True).head(9)

    def safe_read_image(path):
        # Try direct path first
        if os.path.exists(path):
            return plt.imread(path)

        # If stored paths are relative-ish, try joining with known roots
        for root in [IMG_ROOT_1, IMG_ROOT_2, PROJECT_ROOT]:
            p2 = os.path.join(root, path)
            if os.path.exists(p2):
                return plt.imread(p2)

        return None

    def make_grid(df_sel, title, out_name):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()

        for ax, (_, row) in zip(axes, df_sel.iterrows()):
            img = safe_read_image(row["image_path"])
            ax.axis("off")
            if img is None:
                ax.set_title("Missing", fontsize=8)
                continue

            ax.imshow(img)
            ax.set_title(f"{row['frame_type']} | {row['recon_error']:.4f}", fontsize=8)

        plt.suptitle(title)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, out_name)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("✅ Saved:", out_path)

    make_grid(top_anom, "Most Anomalous Frames (Top 9)", "most_anomalous_grid.png")
    make_grid(top_norm, "Most Normal Frames (Bottom 9)", "most_normal_grid.png")

else:
    print("⚠️ vae_recon_scores.csv not found, skipping grids.")
