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
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "UCI HAR Dataset"
DATA_URL = (
    "https://uci.edu"
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

    candidates = [DATA_DIR]
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
            raise RuntimeError("Downloaded file is not a valid ZIP archive.")

        print("Download complete. Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(ROOT)
        print("Dataset extraction finished.")

    except Exception as exc:
        raise RuntimeError(f"Could not download or extract dataset: {exc}") from exc
    finally:
        if zip_path.exists():
            zip_path.unlink()

    dataset = find_dataset()
    if dataset is None:
        raise RuntimeError("Required files could not be located after extraction.")
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
    X = pd.read_csv(split_dir / f"X_{split}.txt", sep=r"\s+", header=None)
    y = pd.read_csv(split_dir / f"y_{split}.txt", sep=r"\s+", header=None).iloc[:, 0]
    subjects = pd.read_csv(split_dir / f"subject_{split}.txt", sep=r"\s+", header=None).iloc[:, 0]

    # Cleaned syntax formatting for feature index maps
    features = pd.read_csv(data_dir / "features.txt", sep=r"\s+", header=None, usecols=[1]).iloc[:, 0].astype(str).tolist()
    features = make_unique_names(features)

    if X.shape[1] != len(features):
        raise ValueError("Feature mismatch with features.txt")

    X.columns = features
    y = y.map(ACTIVITY_NAMES)
    return X, y, subjects


def create_models(seed: int):
    """Create baseline and scaled machine-learning models."""
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1))
        ]),

        "RBF Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(C=4.0, kernel="rbf", class_weight="balanced")),
        ]),
    }


def run_experiment(seed: int = 42):
    """Train models, evaluate performance, and save experiment artifacts."""
    data_dir = ensure_dataset()

    X_train, y_train, subjects_train = load_split("train", data_dir)
    X_test, y_test, subjects_test = load_split("test", data_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 68)
    print("ACTIVITYSENSE | TRAINING RUN ENGINE")
    print("=" * 68)

    result_rows = []
    models = create_models(seed)
    
    best_accuracy = 0.0
    best_model = None

    for name, model in models.items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        acc = float(accuracy_score(y_test, predictions))
        bal_acc = float(balanced_accuracy_score(y_test, predictions))
        macro_f1 = float(f1_score(y_test, predictions, average="macro"))
        weighted_f1 = float(f1_score(y_test, predictions, average="weighted"))

        result_rows.append({
            "model": name,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        })
        
        if acc > best_accuracy and name != "Majority baseline":
            best_accuracy = acc
            best_model = model

    pd.DataFrame(result_rows).to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    bundle = {
        "model": best_model if best_model else models["RBF Support Vector Machine"],
        "feature_names": list(X_train.columns),
    }
    joblib.dump(bundle, MODEL_DIR / "activity_model.joblib")

    summary_data = {
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train.shape[1],
        "selected_model": "RBF Support Vector Machine" if best_model is None else [k for k, v in models.items() if v == best_model][0],
        "limitations": [
            "Dataset collected under strict laboratory environments.",
            "Features pre-engineered from specific sensor windows.",
            "Not tested on raw mobile stream signals."
        ]
    }
    
    with open(OUTPUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print("\n✅ Training complete! Artifacts saved successfully.")

if __name__ == "__main__":
    run_experiment()
