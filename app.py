
from __future__ import annotations

import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "UCI HAR Dataset"
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"

ACTIVITY_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def find_dataset() -> Path | None:
    """Find a valid extracted UCI HAR dataset anywhere inside the project."""

    required_files = [
        "train/X_train.txt",
        "train/y_train.txt",
        "train/subject_train.txt",
        "test/X_test.txt",
        "test/y_test.txt",
        "test/subject_test.txt",
        "features.txt",
    ]

    # Check the expected directory first.
    candidates = [DATA_DIR]

    # Also check other extracted folders within the project.
    for feature_file in ROOT.rglob("features.txt"):
        candidates.append(feature_file.parent)

    checked = set()

    for candidate in candidates:
        candidate = candidate.resolve()

        if candidate in checked or not candidate.is_dir():
            continue

        checked.add(candidate)

        if all((candidate / name).is_file() for name in required_files):
            return candidate

    return None


def ensure_dataset() -> Path:
    """Download, extract, and locate the official UCI HAR dataset."""

    existing_dataset = find_dataset()

    if existing_dataset is not None:
        print(f"Dataset found at: {existing_dataset}")
        return existing_dataset

    zip_path = ROOT / "uci_har_download.zip"

    print("UCI HAR dataset not found. Downloading the official dataset...")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "ActivitySense-HAR/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with open(zip_path, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    output.write(chunk)

        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError(
                "The downloaded file is not a valid ZIP archive. "
                "Delete any incomplete download and run app.py again."
            )

        print("Download complete.")
        print("Extracting dataset. Please wait...")

        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(ROOT)

        print("Dataset extraction finished.")

    except Exception as exc:
        raise RuntimeError(
            "Could not download or extract the UCI HAR dataset. "
            "Check your internet connection and available disk space. "
            f"Details: {exc}"
        ) from exc

    finally:
        if zip_path.exists():
            zip_path.unlink()

    dataset = find_dataset()

    if dataset is None:
        raise RuntimeError(
            "The dataset was downloaded and extracted, but the required "
            "files could not be located.\n"
            "Expected files include train/X_train.txt, test/X_test.txt, "
            "and features.txt.\n"
            f"Project directory: {ROOT}"
        )

    print(f"Dataset successfully located at: {dataset}")

    return dataset


def make_unique_names(names: list[str]) -> list[str]:
    """Make duplicate feature names unique without dropping any features."""
    counts: dict[str, int] = {}
    unique_names: list[str] = []

    for name in names:
        count = counts.get(name, 0)
        counts[name] = count + 1
        unique_names.append(name if count == 0 else f"{name}__{count}")

    return unique_names


def load_split(split: str, data_dir: Path):
    """Load features, activity labels, and participant IDs."""

    split_dir = data_dir / split

    X = pd.read_csv(
        split_dir / f"X_{split}.txt",
        sep=r"\s+",
        header=None,
    )

    y = pd.read_csv(
        split_dir / f"y_{split}.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    subjects = pd.read_csv(
        split_dir / f"subject_{split}.txt",
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    features = pd.read_csv(
        data_dir / "features.txt",
        sep=r"\s+",
        header=None,
        usecols=[1],
    ).iloc[:, 0].astype(str).tolist()

    # UCI HAR contains repeated names for some feature columns.
    # Keep all 561 columns but give every column a unique name.
    features = make_unique_names(features)

    if X.shape[1] != len(features):
        raise ValueError(
            f"Feature mismatch: data has {X.shape[1]} columns, "
            f"but features.txt contains {len(features)} feature names."
        )

    if len(X) != len(y) or len(X) != len(subjects):
        raise ValueError(
            f"The {split} split has inconsistent feature, label, "
            "and subject counts."
        )

    X.columns = features

    y = y.map(ACTIVITY_NAMES)

    if y.isna().any():
        raise ValueError("Unknown activity labels were found in the dataset.")

    return X, y, subjects


def create_models(seed: int):
    """Create baseline and machine-learning models."""

    return {
        "Majority baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),

        "RBF Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    C=4.0,
                    kernel="rbf",
                    class_weight="balanced",
                ),
            ),
        ]),
    }


def run_experiment(seed: int = 42):
    """Train models, evaluate performance, and save experiment artifacts."""

    data_dir = ensure_dataset()

    X_train, y_train, subjects_train = load_split("train", data_dir)
    X_test, y_test, subjects_test = load_split("test", data_dir)

    # Verify that the provided train/test participant split is independent.
    overlap = set(subjects_train.unique()) & set(subjects_test.unique())

    if overlap:
        raise RuntimeError(
            "Train/test subject overlap detected: "
            f"{sorted(overlap)}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("ACTIVITYSENSE | HUMAN ACTIVITY RECOGNITION")
    print("=" * 68)

    print(f"Training examples: {len(X_train):,}")
    print(f"Testing examples:  {len(X_test):,}")
    print(f"Features per example: {X_train.shape[1]}")

    print(
        f"Training participants: {subjects_train.nunique()} | "
        f"Testing participants: {subjects_test.nunique()}"
    )

    print(
        "Activities: "
        + ", ".join(sorted(ACTIVITY_NAMES.values()))
    )

    print("\nTraining and evaluating models...\n")

    result_rows = []
    fitted_models = {}
    reports = {}

    labels = sorted(ACTIVITY_NAMES.values())

    for name, model in create_models(seed).items():
        print(f"Training {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        row = {
            "model": name,
            "accuracy": float(
                accuracy_score(y_test, predictions)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "macro_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="macro",
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                )
            ),
        }

        result_rows.append(row)
        fitted_models[name] = model

        reports[name] = classification_report(
            y_test,
            predictions,
            labels=labels,
            output_dict=True,
            zero_division=0,
        )

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=labels,
        )

        confusion_df = pd.DataFrame(
            matrix,
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
        )

        safe_name = (
            name.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        confusion_df.to_csv(
            OUTPUT_DIR / f"confusion_{safe_name}.csv"
        )

        print(
            f"{name}: "
            f"accuracy={row['accuracy']:.4f}, "
            f"balanced_accuracy={row['balanced_accuracy']:.4f}, "
            f"macro-F1={row['macro_f1']:.4f}\n"
        )

    results = pd.DataFrame(result_rows).sort_values(
        "macro_f1",
        ascending=False,
    )

    results.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "classification_reports.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(reports, file, indent=2)

    # Select the best model using macro-F1.
    best_name = str(results.iloc[0]["model"])
    best_model = fitted_models[best_name]

    best_predictions = best_model.predict(X_test)

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": best_predictions,
    }).to_csv(
        OUTPUT_DIR / "test_predictions.csv",
        index=False,
    )

    # Save the trained model and metadata for dashboard.py.
    model_bundle = {
        "model": best_model,
        "model_name": best_name,
        "feature_names": list(X_train.columns),
        "activity_names": labels,
        "dataset": "UCI HAR",
        "warning": (
            "The standard UCI HAR dataset does not contain a FALL class."
        ),
    }

    model_path = MODEL_DIR / "activity_model.joblib"

    joblib.dump(model_bundle, model_path)

    summary = {
        "project": "ActivitySense Human Activity Recognition",
        "dataset": (
            "UCI Human Activity Recognition Using Smartphones"
        ),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "features": int(X_train.shape[1]),
        "train_subjects": int(subjects_train.nunique()),
        "test_subjects": int(subjects_test.nunique()),
        "selected_model": best_name,
        "selection_metric": "macro_f1",
        "results": results.to_dict(orient="records"),
        "fall_detection_available": False,
        "limitations": [
            "The UCI HAR dataset contains six activities and no fall label.",
            "This experiment uses pre-engineered features, not live phone sensor streams.",
            "Performance on this dataset does not establish clinical safety.",
        ],
    }

    with open(
        OUTPUT_DIR / "experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 68)
    print("MODEL COMPARISON")
    print("=" * 68)

    print(
        results.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {model_path}")
    print(f"Experiment results: {OUTPUT_DIR}")

    print(
        "\nImportant: The standard UCI HAR dataset does not include "
        "a FALL activity."
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Download the UCI HAR dataset, train activity "
            "recognition models, and compare their performance."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    run_experiment(seed=args.seed)


if __name__ == "__main__":
    main()
