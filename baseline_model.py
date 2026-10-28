
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "results" / "shared_split_manifest.csv"
RESULTS_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
AMBIENT_MODEL_PATH = MODEL_DIR / "ambient_logreg_v1.joblib"


def extract_features(image_path):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (128, 128))

    # Six colour features: mean and standard deviation
    # for each of the three colour channels.
    colour_features = []
    for channel in cv2.split(image):
        colour_features.extend([
            float(np.mean(channel)),
            float(np.std(channel)),
        ])

    # Two texture features.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    texture_variance = float(np.var(laplacian))

    return colour_features + [edge_density, texture_variance]


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    manifest = pd.read_csv(MANIFEST)

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "validation"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    def features_for(df):
        paths = [ROOT / path for path in df["path"]]
        X = np.array([extract_features(path) for path in paths])
        y = df["label"].to_numpy(dtype=int)
        return X, y

    X_train, y_train = features_for(train_df)
    X_test, y_test = features_for(test_df)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, AMBIENT_MODEL_PATH)
    print(f"Saved ambient model: {AMBIENT_MODEL_PATH}")
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["untreated", "starch_treated"],
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    )

    print("Logistic Regression — SHARED SPLIT")
    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )
    print("\nTest classification report:")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)

    (RESULTS_DIR / "baseline_test_report.txt").write_text(
        report + "\nConfusion matrix:\n" + str(matrix)
    )

    print("\nSaved report: results/baseline_test_report.txt")


if __name__ == "__main__":
    main()
