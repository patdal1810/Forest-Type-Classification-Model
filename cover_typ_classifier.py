#!/usr/bin/env python3
"""
Cover Type Classification (Beginner-Friendly, Step-by-Step)

What this script does:
1) Loads the forest cover dataset (CSV).
2) Splits rows into Train / Validation / Test sets (so we can evaluate fairly).
3) Preprocesses features:
   - Scales numeric columns (mean=0, std=1) because neural nets learn better when numeric
     values are on similar scales.
   - Leaves binary columns (Soil_Type*, Wilderness_Area*) as-is.
4) Builds a small Deep Learning model (Keras/TensorFlow).
5) Trains with EarlyStopping (stops when val accuracy stops improving).
6) Evaluates on the held-out Test set and prints a confusion matrix + classification report.
7) Saves the model to disk so you can reuse it without retraining.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ------------------------- 1. CONFIG -------------------------
CSV_PATH = os.getenv("CSV_PATH", "cover_data.csv")     # change if your CSV sits elsewhere
RANDOM_SEED = 42
TEST_SIZE   = 0.15   # final, untouched test set
VAL_SIZE    = 0.15   # from the remaining train split, used during training for early stopping

# Map integer class labels (1..7) to human-friendly names
CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

# ------------------------- 2. LOAD DATA -------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}. "
                                f"Pass CSV_PATH=/your/path.csv or move the file next to this script.")
    df = pd.read_csv(csv_path)
    # Some public copies of this dataset call the label "Cover_Type". Your file uses "class".
    if "class" in df.columns:
        label_col = "class"
    elif "Cover_Type" in df.columns:
        label_col = "Cover_Type"
    else:
        raise ValueError("Could not find the target column. Expected 'class' or 'Cover_Type'.")
    return df, label_col

# ------------------------- 3. FEATURE SPLIT -------------------------
def split_features_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[label_col].values.astype(np.int64)
    X = df.drop(columns=[label_col])
    return X, y

def list_numeric_and_binary_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # Heuristic for this dataset:
    # * Numeric columns are the first 10 or so continuous features (Elevation, Aspect, Slope, Distances, Hillshades).
    # * Binary columns start with 'Wilderness_Area' or 'Soil_Type' (0/1 dummies).
    num_cols = []
    bin_cols = []
    for c in X.columns:
        if c.startswith("Wilderness_Area") or c.startswith("Soil_Type"):
            bin_cols.append(c)
        else:
            num_cols.append(c)
    return num_cols, bin_cols

# ------------------------- 4. TRAIN/VAL/TEST SPLIT -------------------------
def make_splits(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # First carve out the test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    # From the remainder, carve out a validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_trainval
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# ------------------------- 5. PREPROCESSING -------------------------
class Preprocessor:
    """
    Scales numeric columns with StandardScaler and concatenates with untouched binary columns.
    We fit on train only, then transform val and test with the learned stats.
    """
    def __init__(self, num_cols: List[str], bin_cols: List[str]):
        self.num_cols = num_cols
        self.bin_cols = bin_cols
        self.scaler = StandardScaler()

    def fit(self, X_train: pd.DataFrame):
        self.scaler.fit(X_train[self.num_cols])

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_num = self.scaler.transform(X[self.num_cols])
        X_bin = X[self.bin_cols].values.astype(np.float32)
        return np.hstack([X_num, X_bin]).astype(np.float32)

# ------------------------- 6. MODEL -------------------------
def build_model(input_dim: int, num_classes: int = 7) -> keras.Model:
    """
    Simple but effective feedforward network.
    You can tune width, depth, dropout, and learning rate later.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ------------------------- 7. TRAIN -------------------------
def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=callbacks,
        verbose=2
    )
    return history

# ------------------------- 8. EVALUATE -------------------------
def evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    print("\\n--- Test Set Performance ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1) + 1  # classes are 1..7
    print("\\nClassification Report:")
    # Map to names for readability in the report
    target_names = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES.keys())]
    print(classification_report(y_test, y_pred, labels=list(CLASS_NAMES.keys()), target_names=target_names))

    print("Confusion Matrix: (rows=true, cols=pred)")
    print(confusion_matrix(y_test, y_pred, labels=list(CLASS_NAMES.keys())))

# ------------------------- 9. (OPTIONAL) SIMPLE HYPERPARAM SEARCH -------------------------
def small_hparam_search(X_train, y_train, X_val, y_val, input_dim):
    """
    A tiny, fast search over two knobs: width and dropout.
    This is NOT exhaustive, but shows the idea.
    """
    configs = [
        {"width": 128, "dropout": 0.2},
        {"width": 256, "dropout": 0.2},
        {"width": 128, "dropout": 0.3},
    ]
    best_acc = -1.0
    best_model = None
    for cfg in configs:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(cfg["width"], activation="relu"),
            layers.Dropout(cfg["dropout"]),
            layers.Dense(cfg["width"]//2, activation="relu"),
            layers.Dense(7, activation="softmax"),
        ])
        model.compile(optimizer=keras.optimizers.Adam(1e-3),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        print(f"\\n>>> Trying config: {cfg}")
        history = model.fit(
            X_train, y_train, validation_data=(X_val, y_val),
            epochs=20, batch_size=512, verbose=0
        )
        val_acc = max(history.history["val_accuracy"])
        print(f"Val best acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model
    print(f"\\nBest config val acc: {best_acc:.4f}")
    return best_model

# ------------------------- 10. MAIN -------------------------
def main():
    # 2) Load
    df, label_col = load_dataset(CSV_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")
    # 3) X/y
    X, y = split_features_labels(df, label_col)
    # 4) Split
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)
    # 5) Preprocess
    num_cols, bin_cols = list_numeric_and_binary_cols(X)
    print(f"Numeric cols: {len(num_cols)} | Binary cols: {len(bin_cols)}")
    prep = Preprocessor(num_cols, bin_cols)
    prep.fit(X_train)
    X_train_p = prep.transform(X_train)
    X_val_p   = prep.transform(X_val)
    X_test_p  = prep.transform(X_test)

    input_dim = X_train_p.shape[1]

    # 6) Model
    model = build_model(input_dim=input_dim)

    # 7) Train
    history = train_model(model, X_train_p, y_train, X_val_p, y_val)

    # (Optional) swap with a quick hyperparam search
    # model = small_hparam_search(X_train_p, y_train, X_val_p, y_val, input_dim)

    # 8) Evaluate
    evaluate(model, X_test_p, y_test)

    # 11) Save
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "cover_type_keras_model.keras"))
    print(f"Model saved to {os.path.join(model_dir, 'cover_type_keras_model.keras')}")

if __name__ == "__main__":
    main()
