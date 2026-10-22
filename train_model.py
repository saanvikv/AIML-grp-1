import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from app.phishing.feature_extractor import (
    FEATURE_NAMES,
    DATASET_COLUMN_MAP,
    extract_features_from_dataset_row,
)

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "phishing.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "phishing_model.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

LIVE_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "live_url_model.joblib"
)


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


print("Loading dataset...")

df = pd.read_csv(
    DATASET_PATH,
    low_memory=False
)

print(
    f"Dataset rows: {len(df)}"
)

print(
    f"Dataset columns: {len(df.columns)}"
)

print(
    "Checking 50 feature columns..."
)

missing = []

for feature in FEATURE_NAMES:
    column = DATASET_COLUMN_MAP.get(
        feature,
        feature
    )

    if column not in df.columns:
        missing.append(column)

if missing:
    raise RuntimeError(
        "Missing dataset columns: "
        + str(missing)
    )

print(
    "50 feature columns verified."
)


# --------------------------------------------------
# Primary 50-feature dataset
# --------------------------------------------------

X = np.asarray(
    [
        extract_features_from_dataset_row(
            row
        )
        for _, row in df.iterrows()
    ],
    dtype=np.float32
)

y = pd.to_numeric(
    df["label"],
    errors="coerce"
).fillna(0).astype(int).to_numpy()


print(
    "Feature shape:",
    X.shape
)

print(
    "Label distribution:",
    dict(
        zip(
            *np.unique(
                y,
                return_counts=True
            )
        )
    )
)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# --------------------------------------------------
# Primary Keras model
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(50,)
    ),

    tf.keras.layers.Dense(
        256,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.30
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dropout(
        0.25
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.20
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )
])


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


print(
    "\nTraining primary 50-feature model..."
)

model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.20,
    epochs=15,
    batch_size=256,
    callbacks=[early_stopping],
    verbose=1
)


predictions = model.predict(
    X_test_scaled,
    verbose=0
).reshape(-1)

predicted_labels = (
    predictions >= 0.5
).astype(int)


accuracy = accuracy_score(
    y_test,
    predicted_labels
)


print(
    f"\nPrimary Test Accuracy: "
    f"{accuracy * 100:.2f}%"
)


joblib.dump(
    scaler,
    SCALER_PATH
)

model.save(
    MODEL_PATH
)


print(
    "\nSaved:",
    MODEL_PATH
)

print(
    "Saved:",
    SCALER_PATH
)

print(
    "Model input:",
    model.input_shape[-1]
)

print(
    "Scaler input:",
    scaler.n_features_in_
)


# --------------------------------------------------
# Secondary live URL model
# --------------------------------------------------

print(
    "\nTraining live URL model..."
)

urls = (
    df["URL"]
    .astype(str)
    .fillna("")
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=30000,
    sublinear_tf=True
)

url_vectors = vectorizer.fit_transform(
    urls
)


live_classifier = SGDClassifier(
    loss="log_loss",
    class_weight="balanced",
    random_state=42,
    max_iter=1000,
    tol=1e-3
)


live_classifier.fit(
    url_vectors,
    y
)


live_model = {
    "vectorizer": vectorizer,
    "classifier": live_classifier,
}


joblib.dump(
    live_model,
    LIVE_MODEL_PATH
)


print(
    "Saved live URL model:",
    LIVE_MODEL_PATH
)

print(
    "\nTraining complete."
)