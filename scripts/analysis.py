
"""
analysis script.
Reads combined predictions CSV and makes basic validation plots per model
Detects models by columns named pred_<model> and uses prob_<model> if present
Writes one PNG per plot into --save_dir as <model>_roc.png, <model>_pr.png,
<model>_probs_hist.png.

Usage:
  python analysis.py --pred_file ../results/combined_predictions.csv \
    --save_dir ../results
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)


def parse_args():
    p = argparse.ArgumentParser(description="Very small prediction analysis")
    p.add_argument("--pred_file", required=True, help="CSV with combined preds")
    p.add_argument("--save_dir", required=True, help="directory to save PNGs")
    return p.parse_args()


def detect_models(df):
    models = []
    for c in df.columns:
        if c.startswith("pred_"):
            name = c[len("pred_") :]
            if name:
                models.append(name)
    return models

def print_confusion(name, y_true, preds):
    """ print a tiny binary confusion matrix """
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    total = int(tn + fp + fn + tp)
    print(f"[confusion] {name}  total={total}")
    print(f" TN={int(tn):3d} | FP={int(fp):3d}")
    print(f" FN={int(fn):3d} | TP={int(tp):3d}")


def save_roc(name, y_true, probs, outdir):
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"ROC {name} AUC={roc_auc:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_roc.png"))
    plt.close()


def save_pr(name, y_true, probs, outdir):
    precision, recall, _ = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.title(f"PR {name} AP={ap:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_pr.png"))
    plt.close()


def save_prob_hist(name, y_true, probs, outdir):
    pos = probs[y_true == 1]
    neg = probs[y_true == 0]
    plt.figure()
    plt.hist(pos, bins=20, alpha=0.6)
    plt.hist(neg, bins=20, alpha=0.6)
    plt.xlabel("predicted probability")
    plt.title(f"Prob dist {name} (pos n={len(pos)} neg n={len(neg)})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{name}_probs_hist.png"))
    plt.close()


def save_combined_roc(models, y_true, probs_map, outdir):
    plt.figure()
    for name in models:
        probs = probs_map[name]
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title("Combined ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "combined_roc.png"))
    plt.close()


def main():
    args = parse_args()
    df = pd.read_csv(args.pred_file)
    
    # create sub folder for plots
    os.makedirs(args.save_dir, exist_ok=True)
    plots_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # could be made adjustable. choose to be simple for demo
    y_series = df['cancer_status']
    y_true = y_series.to_numpy()

    # detects which models were used via pred_<model> e.g. pred_rf, pred_nn
    models = detect_models(df)
    if not models:
        raise RuntimeError("no pred_<model> columns found in file")

    probs_map = {}
    for name in models:
        pred_col = f"pred_{name}"
        prob_col = f"prob_{name}"
        preds = df[pred_col].to_numpy()
        if prob_col in df.columns:
            probs = df[prob_col].to_numpy()
        else:
            probs = preds

        probs_map[name] = probs

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        print(f"[analysis] {name} acc={acc:.3f} f1={f1:.3f}")

        save_roc(name, y_true, probs, plots_dir)
        save_pr(name, y_true, probs, plots_dir)
        save_prob_hist(name, y_true, probs, plots_dir)
        print_confusion(name, y_true, preds)

    # combined ROC for all detected models
    save_combined_roc(models, y_true, probs_map, plots_dir)
    print("[analysis] done. plots saved to", plots_dir)


if __name__ == "__main__":
    main()
