# evaluate_results.py
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)

# ==== CONFIG ====
RESULTS_DIR = r"D:\important\projects\Frog\evaluation"
OUTPUT_DIR = RESULTS_DIR
THRESHOLDS = np.arange(0.0, 1.01, 0.05)

def load_results(folder):
    """Load the BirdNET CombinedTable results only."""
    fpath = os.path.join(folder, "BirdNET_CombinedTable.csv")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"No CombinedTable found at {fpath}")
    return pd.read_csv(fpath)

def get_labels_from_path(path: str):
    fname = os.path.basename(path).lower()
    parts = fname.replace(".wav", "").split("_")

    if "positive" in parts:
        true = "RADR"
        idx = parts.index("positive")
        quality = parts[idx+1] if idx+1 < len(parts) else "unknown"
        calltype = parts[idx+2] if idx+2 < len(parts) else "unknown"
    elif "negative" in parts:
        true, quality, calltype = "Negative", "negative", "negative"
    else:
        # fallback, shouldn't really happen
        true, quality, calltype = "Negative", "unknown", "unknown"

    return true, quality, calltype




def sanity_check_labels(df):
    """Print counts of parsed qualities and calltypes for debugging."""
    q_counts = Counter(df["quality"])
    c_counts = Counter(df["calltype"])
    print("Quality distribution:", dict(q_counts))
    print("Calltype distribution:", dict(c_counts))


def evaluate(df, split_name):
    """Evaluate performance across thresholds + by group."""
    # Prepare labels
    df["true_label"], df["quality"], df["calltype"] = zip(*df["File"].map(get_labels_from_path))
    df["pred_label"] = np.where(df["Scientific name"] == "RADR", "RADR", "Negative")
    df["score"] = df["Confidence"]

    y_true = (df["true_label"] == "RADR").astype(int).values
    y_score = df["score"].values

        # ===== Threshold sweep =====
    rows = []
    for thr in THRESHOLDS:
        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1,0])
        tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)

        precision = tp / (tp + fp) if (tp+fp) else 0
        recall = tp / (tp + fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
        fprate = fp / (fp + tn) if (fp+tn) else 0   # new
        tnr = tn / (tn + fp) if (tn+fp) else 0      # new

        rows.append({
            "split": split_name,
            "threshold": thr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": acc,
            "fpr": fprate,        # new column
            "tnr": tnr,           # new column
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })

    metrics_df = pd.DataFrame(rows)


    # ===== Curves =====
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, label=f"{split_name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, f"roc_curve_{split_name}.png"))
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"{split_name} (AP={pr_auc:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR Curve")
    plt.legend(); plt.savefig(os.path.join(OUTPUT_DIR, f"pr_curve_{split_name}.png"))
    plt.close()

    # ===== Group breakdown (skip negatives) =====
    group_rows = []
    for group_col in ["quality", "calltype"]:
        for group, sub in df.groupby(group_col):
            if group == "negative":   # skip negatives
                continue
            y_true_g = (sub["true_label"]=="RADR").astype(int)
            y_pred_g = (sub["pred_label"]=="RADR").astype(int)
            if len(y_true_g)==0: 
                continue
            cm = confusion_matrix(y_true_g, y_pred_g, labels=[1,0])
            tp, fn, fp, tn = cm.ravel()
            precision = tp/(tp+fp) if (tp+fp) else 0
            recall = tp/(tp+fn) if (tp+fn) else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
            acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
            group_rows.append({
                "split": split_name,
                "group_by": group_col,
                "group": group,
                "n": len(sub),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": acc,
                "tp": tp,"fp":fp,"fn":fn,"tn":tn
            })

    return metrics_df, pd.DataFrame(group_rows)


def main():
    all_metrics, all_groups = [], []

    for split in ["TestIID","TestOOD"]:
        folder = os.path.join(RESULTS_DIR, split)
        df = load_results(folder)
        metrics_df, groups_df = evaluate(df, split)
        all_metrics.append(metrics_df)
        all_groups.append(groups_df)

    pd.concat(all_metrics).to_csv(os.path.join(OUTPUT_DIR,"metrics_summary.csv"), index=False)
    pd.concat(all_groups).to_csv(os.path.join(OUTPUT_DIR,"metrics_by_group.csv"), index=False)
    print("✅ Metrics written to CSV, plots saved to evaluation folder.")

if __name__=="__main__":
    main()
