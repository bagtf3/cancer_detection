"""
minimal predictor that loads multiple model subdirs.

 - config JSON contains "features": [..] list of column names.
 - model_dir contains subdirectories named after models during training (e.g. rf, nn)
     - rf subdir contains rfmodel.joblib
     - nn subdir contains scaler.joblib + keras_model.h5

Usage example:
python predict.py \
  --model_dir ../results \
  --models rf,nn \
  --input_file ../data/cancer_data_train.csv \
  --config ../configs/config_simple.json \
  --out_csv ../results/combined_predictions.csv
"""

import os
import json
import argparse
import pandas as pd

from cancer_detection.model import get_model
from cancer_detection.utils import load_config


def parse_args():
    p = argparse.ArgumentParser(
        description="Load models from model_dir/<name> and predict"
    )

    p.add_argument(
        "--model_dir", required=True,
        help="directory containing model subdirs (rf, nn, ...)"
    )

    p.add_argument(
        "--models", required=True,
        help="comma-separated models to load, e.g. rf,nn"
    )
    
    p.add_argument("--input_file", required=True, help="CSV with feature columns")
    p.add_argument("--config", required=True, help="config JSON with 'features' list")
    p.add_argument(
        "--out_csv", required=True, help="output CSV path (combined predictions)"
    )

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    features = cfg["features"]

    df = pd.read_csv(args.input_file)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError("Missing feature columns in input CSV: " + str(missing))

    X = df[features].to_numpy()

    models_list = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    out = df.copy().reset_index(drop=False)

    # preserve original index
    out = out.rename(columns={"index": "orig_index"})

    for m in models_list:
        subdir = os.path.join(args.model_dir, m)
        print(f"[predict] loading model '{m}' from {subdir}")

        # unbuilt model object
        model = get_model(m)

        # load/build with the saved model from training
        model.load(subdir)

        print(f"[predict] predicting with '{m}'")
        preds = model.predict(X)
        probs = model.predict_proba(X)

        out[f"pred_{m}"] = preds
        out[f"prob_{m}"] = probs

    out_dir = os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print("[predict] wrote ->", args.out_csv)

if __name__ == "__main__":
    main()