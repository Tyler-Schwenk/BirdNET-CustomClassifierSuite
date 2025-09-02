# scripts/evaluate_results.py
import os
import pandas as pd
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

THRESHOLDS = np.arange(0.0, 1.01, 0.05)

def make_experiment_summary(exp_dir: str):
    eval_dir = os.path.join(exp_dir, "evaluation")
    metrics_path = os.path.join(eval_dir, "metrics_summary.csv")
    config_path = os.path.join(exp_dir, "config_used.yaml")
    data_summary_path = os.path.join(exp_dir, "training_package", "data_summary.csv")
    selection_report_path = os.path.join(exp_dir, "training_package", "selection_report.json")

    if not os.path.exists(metrics_path) or not os.path.exists(config_path):
        print("Cannot build experiment_summary.json — missing metrics or config.")
        return

    # Load metrics + config
    metrics = pd.read_csv(metrics_path)
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Load selection report if available
    selection_info, metadata = {}, {}
    if os.path.exists(selection_report_path):
        with open(selection_report_path, "r") as f:
            selection_info = json.load(f)
        metadata = selection_info.get("metadata", {})

    # Load dataset breakdowns
    breakdown_quality, breakdown_calltype = {}, {}
    neg_total = None
    if os.path.exists(data_summary_path):
        ds = pd.read_csv(data_summary_path)
        ds_sel = ds[ds["stage"] == "selected"]

        pos_quality = ds_sel[(ds_sel["label"] == "positive") & (ds_sel["group_by"] == "quality")]
        breakdown_quality = dict(zip(pos_quality["group"], pos_quality["count"]))

        pos_call = ds_sel[(ds_sel["label"] == "positive") & (ds_sel["group_by"] == "call_type")]
        breakdown_calltype = dict(zip(pos_call["group"], pos_call["count"]))

        neg_sel = ds_sel[(ds_sel["label"] == "negative") & (ds_sel["group_by"] == "split")]
        neg_total = int(neg_sel["count"].sum()) if not neg_sel.empty else 0

    def clean_row(row_dict):
        """Keep only essential fields, round floats."""
        keys = ["threshold", "precision", "recall", "f1", "accuracy", "tp", "fp", "fn", "tn"]
        clean = {}
        for k in keys:
            if k in row_dict:
                val = row_dict[k]
                if isinstance(val, (float, np.floating)):
                    clean[k] = round(float(val), 3)
                else:
                    clean[k] = int(val)
        return clean

    def pick_metrics(df, split):
        """Return dict of metrics at default=0.5, best F1, high precision + AUROC/AUPRC."""
        sub = df[df["split"] == split]
        if sub.empty:
            return {}

        # Default at threshold 0.5
        def_row = clean_row(sub.loc[(sub["threshold"] - 0.5).abs().idxmin()].to_dict())

        # Best F1
        f1_row = clean_row(sub.loc[sub["f1"].idxmax()].to_dict())

        # Highest threshold with precision >= 0.9
        highp = sub[sub["precision"] >= 0.9]
        hp_row = clean_row(highp.loc[highp["threshold"].idxmax()].to_dict()) if not highp.empty else None

        # Pull AUROC and AUPRC
        auc_path = os.path.join(eval_dir, f"summary_{split}.json")
        auroc, auprc = None, None
        if os.path.exists(auc_path):
            with open(auc_path, "r") as f:
                auc_info = json.load(f)
                auroc = round(auc_info.get("roc_auc", 0.0), 3)
                auprc = round(auc_info.get("pr_auc", 0.0), 3)

        return {
            "default_0.5": def_row,
            "best_f1": f1_row,
            "high_precision": hp_row,
            "auroc": auroc,
            "auprc": auprc
        }

    summary = {
        "experiment": cfg.get("experiment", {}),
        "training": cfg.get("training", {}),
        "dataset": {
            "manifest": cfg.get("dataset", {}).get("manifest"),
            "filters": selection_info.get("filters", {}),
            "counts_selected": selection_info.get("counts_selected", {}),
            "breakdown_selected": {
                "positive_quality": breakdown_quality,
                "positive_call_type": breakdown_calltype,
                "negative_total": neg_total
            }
        },
        "metrics": {
            "iid": pick_metrics(metrics, "test_iid"),
            "ood": pick_metrics(metrics, "test_ood")
        },
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "git_commit": metadata.get("git_commit"),
            "hostname": metadata.get("hostname"),
            "user": metadata.get("user")
        }
    }

    outpath = os.path.join(eval_dir, "experiment_summary.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {outpath}")

def load_results(folder):
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
        true, quality, calltype = "Negative", "unknown", "unknown"
    return true, quality, calltype

def evaluate(df, split_name, outdir):
    df["true_label"], df["quality"], df["calltype"] = zip(*df["File"].map(get_labels_from_path))
    df["pred_label"] = np.where(df["Scientific name"] == "RADR", "RADR", "Negative")
    df["score"] = df["Confidence"]

    y_true = (df["true_label"] == "RADR").astype(int).values
    y_score = df["score"].values

    # === Threshold sweep ===
    rows = []
    for thr in THRESHOLDS:
        y_pred = (y_score >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
        precision = tp / (tp + fp) if (tp+fp) else 0
        recall = tp / (tp + fn) if (tp+fn) else 0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
        acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
        fpr = fp / (fp + tn) if (fp+tn) else 0
        tnr = tn / (tn + fp) if (tn+fp) else 0
        rows.append({
            "split": split_name, "threshold": thr,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": acc, "fpr": fpr, "tnr": tnr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn
        })
    metrics_df = pd.DataFrame(rows)

    # === Curves ===
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    # Save curve data
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(os.path.join(outdir, f"roc_data_{split_name}.csv"), index=False)
    pd.DataFrame({"recall": recall, "precision": precision}).to_csv(os.path.join(outdir, f"pr_data_{split_name}.csv"), index=False)

    # Save plots
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],"k--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({split_name})"); plt.legend()
    plt.savefig(os.path.join(outdir, f"roc_curve_{split_name}.png")); plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR Curve ({split_name})"); plt.legend()
    plt.savefig(os.path.join(outdir, f"pr_curve_{split_name}.png")); plt.close()

    # === Group breakdown ===
    group_rows = []
    for group_col in ["quality", "calltype"]:
        for group, sub in df.groupby(group_col):
            if group == "negative":
                continue
            y_true_g = (sub["true_label"]=="RADR").astype(int)
            y_pred_g = (sub["pred_label"]=="RADR").astype(int)
            if len(y_true_g) == 0:
                continue
            cm = confusion_matrix(y_true_g, y_pred_g, labels=[1,0])
            tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
            precision = tp/(tp+fp) if (tp+fp) else 0
            recall = tp/(tp+fn) if (tp+fn) else 0
            f1 = 2*precision*recall/(precision+recall) if (precision+recall) else 0
            acc = (tp+tn)/(tp+tn+fp+fn) if (tp+tn+fp+fn) else 0
            group_rows.append({
                "split": split_name, "group_by": group_col, "group": group,
                "n": len(sub),
                "precision": precision, "recall": recall, "f1": f1, "accuracy": acc,
                "tp": tp,"fp":fp,"fn":fn,"tn":tn
            })
    groups_df = pd.DataFrame(group_rows)

    # === Snapshot JSON (quick summary) ===
    summary = {
        "split": split_name,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "n_samples": len(df)
    }
    pd.Series(summary).to_json(os.path.join(outdir, f"summary_{split_name}.json"))

    return metrics_df, groups_df

def run_evaluation(exp_dir: str):
    """Run evaluation for a given experiment dir."""
    inf_dir = os.path.join(exp_dir, "inference")
    out_dir = os.path.join(exp_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    all_metrics, all_groups = [], []
    for split in ["test_iid", "test_ood"]:
        folder = os.path.join(inf_dir, split)
        df = load_results(folder)
        metrics_df, groups_df = evaluate(df, split, out_dir)
        all_metrics.append(metrics_df)
        all_groups.append(groups_df)

    pd.concat(all_metrics).to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
    pd.concat(all_groups).to_csv(os.path.join(out_dir, "metrics_by_group.csv"), index=False)
    make_experiment_summary(exp_dir)
    print(f"Evaluation complete. Metrics written to {out_dir}")
