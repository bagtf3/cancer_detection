"""
Minimal multi model predictor
Loads models from model_dir/<name> and predicts using features from config
Writes one CSV with pred_<model> and prob_<model>

Usage:
  python predict.py --model_dir ../results --models rf,nn \
    --input_file ../data/cancer_data_train.csv \
    --config ../configs/config_simple.json \
    --out_csv ../results/combined_predictions.csv
"""


import os
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

from cancer_detection.model import get_model
from cancer_detection.utils import load_config, write_json


def parse_args():
    p = argparse.ArgumentParser(description="Simple train script")
    p.add_argument(
        "--models", required=True,
        help="comma-separated list of models to train, e.g. rf,nn"
    )
    p.add_argument("--input_file", required=True, help="path to csv input data")
    p.add_argument("--config_file", required=True, help="path to JSON config")
    p.add_argument("--results_dir", required=True, help="directory to write outputs")
    return p.parse_args()


def prepare_data(input_file, cfg):
    df = pd.read_csv(input_file)
    features = cfg["features"]
    target = cfg["target"]

    # will raise if any names missing
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise RuntimeError("missing columns in input_file: " + str(missing))

    X = df[features].to_numpy()
    y_raw = df[target]

    # accept numeric 0/1 or treat truthy '1' as positive
    if pd.api.types.is_numeric_dtype(y_raw):
        y = y_raw.to_numpy().astype("int64")
    else:
        y = (y_raw == "1").to_numpy().astype("int64")

    return df, X, y, features


def train_one(name, model_obj, model_cfg, X_train, y_train, X_val, y_val, out_dir):
    print(f"[train] -> {name}")

    # always attempt to call build with input_dim
    model_obj.build(input_dim=X_train.shape[1])

    # pass a handful of possible NN fit kwargs if present
    fit_kwargs = {}
    for k in ("epochs", "batch_size", "verbose", "validation_split"):
        if k in model_cfg:
            fit_kwargs[k] = model_cfg[k]

    model_obj.fit(X_train, y_train, **fit_kwargs)
    print(f"[train] fitted {name}")

    probs = model_obj.predict_proba(X_val)
    preds = model_obj.predict(X_val)

    # basic training metrics
    acc = metrics.accuracy_score(y_val, preds)
    f1 = metrics.f1_score(y_val, preds)
    auc = metrics.roc_auc_score(y_val, probs)

    metrics_obj = {"accuracy": acc, "f1": f1, "auc": auc}
    print(f"[metrics] {name} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")

    # save model flatly into out_dir (no extra nested subdir)
    # out_dir already is .../results/<name>
    os.makedirs(out_dir, exist_ok=True)
    model_obj.save(out_dir)
    print(f"[train] saved model -> {out_dir}")

    # write predictions (still flat under out_dir)
    preds_df = pd.DataFrame({"true": y_val, "pred": preds, "prob": probs})
    preds_csv = os.path.join(out_dir, f"{name}_val_predictions.csv")
    preds_df.to_csv(preds_csv, index=False)
    print(f"[train] wrote preds -> {preds_csv}")

    metrics_path = os.path.join(out_dir, f"{name}_metrics.json")
    write_json(metrics_path, metrics_obj)
    print(f"[train] wrote metrics -> {metrics_path}")


def main():
    args = parse_args()
    models_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    cfg = load_config(args.config_file)

    os.makedirs(args.results_dir, exist_ok=True)
    # config contains meta info for model(s) and data
    df, X, y, features = prepare_data(args.input_file, cfg)

    test_size = cfg.get("test_size", 0.2)
    random_state = cfg.get("random_state", 0)

    stratify = y if len(np.unique(y)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    print(f"[train] data split train={len(X_train)} val={len(X_val)} feats={features}")

    model_configs = cfg.get("models", {})

    for m in models_list:
        # find the matching config for this model
        model_cfg = model_configs.get(m, {}) if isinstance(model_configs, dict) else {}
        print(f"[train] building model '{m}' with cfg: {model_cfg}")

        # returns an unbuilt model
        model_obj = get_model(m, **model_cfg)

        out_dir = os.path.join(args.results_dir, m)
        os.makedirs(out_dir, exist_ok=True)

        # builds the model with the config and input data, trains, saves
        train_one(m, model_obj, model_cfg, X_train, y_train, X_val, y_val, out_dir)

    print("[train] done. results in", args.results_dir)


if __name__ == "__main__":
    main()
