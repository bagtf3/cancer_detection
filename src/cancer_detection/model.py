import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from tensorflow.keras import layers, models


# helper
def to_array(X):
    if hasattr(X, "values"):
        return X.values
    return np.asarray(X)


class RFModel:
    """
    thin RF adapter. Underlying sklearn pipeline is created on
    build() or loaded from disk with load().
    """

    def __init__(self, n_estimators=100, **kwargs):
        self.n_estimators = n_estimators
        self.extra = dict(kwargs)
        self.model = None
        self.built = False
    
    def raise_not_ready(self):
        raise RuntimeError("RFModel not built or loaded. Call build()/load().")
        
    def build(self, input_dim=None):
        params = dict(self.extra)
        params.update({"n_estimators": self.n_estimators})
        
        clf = RandomForestClassifier(**params)
        self.model = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        self.built = True

    def fit(self, X, y):
        if not self.built:
            self.raise_not_ready()
            
        self.model.fit(X, y)

    def predict(self, X):
        if not self.built:
            self.raise_not_ready()
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.built:
            self.raise_not_ready()
        probs = self.model.predict_proba(X)
        return probs[:, 1]

    def save(self, path):
        if not self.built:
            self.raise_not_ready()
        filename = os.path.join(path, "rfmodel.joblib")
        joblib.dump(self.model, filename)

    def load(self, path):
        filename = os.path.join(path, "rfmodel.joblib")
        obj = joblib.load(filename)
        # assume loaded object is pipeline
        self.model = obj
        self.built = True


class NNModel:
    """
    Tiny TF NN adapter. build(input_dim=int) will create a compiled
    keras.Model. If build is omitted, fit will build lazily from X shape.
    """

    def __init__(self, hidden_units=(32, 16), dropout=0.2,
                 optimizer="adam", loss="binary_crossentropy",
                 metrics=None, **kwargs):
        
        self.hidden_units = [h for h in hidden_units]
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics or [tf.keras.metrics.AUC(name="auc")]
        self.extra = dict(kwargs)

        self.model = None
        self.scaler = None
        self.built = False
    
    def raise_not_ready(self):
        raise RuntimeError("NNModel not built or loaded. Call build()/load().")
        
    def build_keras(self, input_dim):
        tf.keras.backend.clear_session()

        inp = layers.Input(shape=(int(input_dim),))
        x = inp
        for h in self.hidden_units:
            x = layers.Dense(int(h), activation="relu")(x)
            if self.dropout and self.dropout > 0.0:
                x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        m = models.Model(inp, out)
        m.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return m

    def build(self, input_dim=None):
        # prepare scaler; building keras model is optional until fit
        self.scaler = StandardScaler()
        if input_dim is not None:
            self.model = self.build_keras(input_dim)
            self.built = True
        else:
            # mark scaler create. model will be built in fit if needed
            self.built = True

    def fit(self, X, y, epochs=50, batch_size=32, verbose=0,
            validation_split=0.1):
        if not self.built:
            self.raise_not_ready()

        # ensure scaler exists
        if self.scaler is None:
            self.scaler = StandardScaler()
            
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)

        # build keras model lazily if absent
        if self.model is None:
            input_dim = Xs.shape[1]
            self.model = self.build_keras(input_dim)

        # simple fit call
        self.model.fit(
            Xs, y, epochs=epochs, batch_size=batch_size,
            validation_split=validation_split, verbose=verbose
        )

    def predict(self, X):
        if not self.built:
            self.raise_not_ready()
            
        if self.model is None:
            self.raise_not_ready()
            
        Xs = self.scaler.transform(X)
        probs = self.model.predict(Xs, verbose=0)
        return (probs.ravel() >= 0.5).astype(int)
    
    # to match sklearn raw probs
    def predict_proba(self, X):
        if not self.built:
            self.raise_not_ready()
        if self.model is None:
            self.raise_not_ready()
            
        Xs = self.scaler.transform(X)
        probs = self.model.predict(Xs, verbose=0)
        return probs.ravel()

    def save(self, path):
        if not self.built:
            self.raise_not_ready()
        
        # save scaler
        scaler_path = os.path.join(path, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # save keras model dir
        keras_path = os.path.join(path, "keras_model.h5")
        if self.model is None:
            self.raise_not_ready()
            
        self.model.save(keras_path)

    def load(self, path):
        scaler_path = os.path.join(path, "scaler.joblib")
        keras_path = os.path.join(path, "keras_model.h5")
        if not os.path.exists(keras_path) or not os.path.exists(scaler_path):
            raise RuntimeError("NNModel load path missing keras_model or scaler.joblib")
            
        self.scaler = joblib.load(scaler_path)
        self.model = tf.keras.models.load_model(keras_path)
        self.built = True


class CustomModel:
    """
    Thin wrapper for a pure math/ c++ model.
    Would have the same look and feel as above.
    """

    def __init__(self, estimator=None, **kwargs):
        pass


def get_model(model_type, **kwargs):
    """
    Return an instance of the chosen model.
    The returned instance is not necessarily 'built' (estimator may be None)
    until you call build() or load().
    """
    t = (model_type or "").lower()
    if t == "rf":
        return RFModel(**kwargs)
    
    if t in ("nn", "tf", "keras"):
        return NNModel(**kwargs)
    
    if t == "custom":
        return CustomModel(**kwargs)
    raise ValueError("unsupported model_type: " + str(model_type))

