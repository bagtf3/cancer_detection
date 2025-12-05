import pandas as pd
import numpy as np
import json


def to_array(X):
    if hasattr(X, "values"):
        return X.values
    return np.asarray(X)


def load_config(path):
    with open(path, "r") as fh:
        return json.load(fh)


def write_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)