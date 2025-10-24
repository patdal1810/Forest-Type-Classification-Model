"""
Cover Type Classification 

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
import matplotlib.pyplot as plt


# -------------- Config ----------------------------

CSV_PATH = os.getenv("CSV_PATH", "cover_data.csv") # path to csv
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# Mapping integer class labels to human-friendly names
CLASS_NAMES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}

target_names = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

# Loading Data
def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find CSV at {csv_path}."
                                f"Pass CSV_PATH=/your/path.csv or move the file next to htis script")

    df = pd.read_csv(csv_path)

    # Some public copies of this dataset call the label "Cover_Type". Your file uses "class".
    if "class" in df.columns:
        label_col = "class"
    elif "Cover_Type" in df.columns:
        label_col = "Cover_Type"
    else:
        raise ValueError("Could not find the target column. Expected 'class' or 'Cover_Type")
    
    return df, label_col


# Lets split the features
def split_features_labels(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, np.ndarray]:
    y = df[label_col].values.astype(np.int64) - 1
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

# Train/ Val / Test Splitting 
def make_splits(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    # Let us carve out the test set first
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    # Let us carve out a remainder from the validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=RANDOM_SEED, stratify=y_trainval
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Let us Preprocessing
class Preprocessor:
    """
    Scales numeric columns with StandardScaler and concatenates with untouched binary columns
    We are going to fit on train set only, then transform val and test with the learned stats.
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
    

# Lets work on the MODEL
def build_model(input_dim: int, num_classes: int = 7) -> keras.Model:
    """
    This will be simple but effective for feedforward network.
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

    # Lets compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-2),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# Time to train the model
def train_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, verbose=1
        )
    ]

    # Fitting the model
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=512,
        callbacks=callbacks,
        verbose=2
    )

    return history

# Evaluating the model
def evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray):
    print("\\n Test Set Performance")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1) # classes are 1..7
    print("\nClassification Report")
    print(classification_report(y_test, y_pred, labels=range(7), target_names=target_names))

    print("Confusion Matrix: (rows=true, cols=pred)")
    print(confusion_matrix(y_test, y_pred, labels=range(7)))



# ================== MAIN =======================
def main():
    # Load dataset
    df, label_col = load_dataset(CSV_PATH)
    print(f"Loaded {df.shape[0]:,} rows, {df.shape[1]} columns.")

    # Split the dataset
    X, y = split_features_labels(df, label_col)
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    # Preprocessing
    num_cols, bin_cols = list_numeric_and_binary_cols(X)
    print(f"Numeric cols: {len(num_cols)} | Binary cols: {len(bin_cols)}")
    prep = Preprocessor(num_cols, bin_cols)
    prep.fit(X_train)
    X_train_p = prep.transform(X_train)
    X_val_p = prep.transform(X_val)
    X_test_p = prep.transform(X_test)

    input_dim = X_train_p.shape[1]


    # Building Model
    model = build_model(input_dim=input_dim)

    # Training the model
    history = train_model(model, X_train_p, y_train, X_val_p, y_val)


    # Plot to see what's goin on
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()

    # Evaluate Model
    evaluate(model, X_test_p, y_test)

    # Saving the model
    model_dir = "saved_model"
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "cover_type_model.keras"))
    print(f"Model saved to {os.path.join(model_dir, 'cover_type_model.keras')}")

if __name__ == "__main__":
    main()