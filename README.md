# ActivitySense: Human Activity Recognition with Smartphone Sensors

**A reproducible machine-learning project for recognizing human activities from smartphone motion data.**

ActivitySense trains and evaluates a majority-class baseline, Random Forest, and RBF Support Vector Machine using the official train/test split of the UCI Human Activity Recognition Using Smartphones dataset. It includes a command-line experiment and a Streamlit dashboard.

## Important dataset note

The standard UCI HAR dataset contains six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying. **It does not contain a FALL class.** This project does not claim validated fall detection. A fall-detection extension requires a separate, appropriate dataset and independent evaluation.

## Features

- Automatically downloads and extracts the public UCI HAR dataset.
- Uses the official participant-separated training and test sets.
- Checks that training and test subject IDs do not overlap.
- Compares a simple baseline, Random Forest, and RBF SVM.
- Reports accuracy, balanced accuracy, macro-F1, and weighted-F1.
- Saves confusion matrices, reports, predictions, experiment summary, and trained model.
- Provides a Streamlit dashboard for viewing results and predicting from correctly formatted feature CSV files.

## Quick start (Windows / VS Code)

Install Python 3.10 or newer. Open PowerShell in this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

The first run downloads the official dataset and may take a little time. An internet connection is required for this initial download.

If PowerShell blocks activation, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

## Launch the dashboard

After `python app.py` finishes:

```powershell
streamlit run dashboard.py
```

Streamlit will display a local address in the terminal; open that address in your browser.

## Project structure

```text
ActivitySense_HAR/
├── app.py
├── dashboard.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── activity_model.joblib       # generated after training
└── outputs/
    ├── model_comparison.csv         # generated after training
    ├── classification_reports.json  # generated after training
    ├── experiment_summary.json      # generated after training
    ├── test_predictions.csv         # generated after training
    └── confusion_*.csv              # generated after training
```

## Methodology

The dataset supplies 561 engineered features per observation. The program uses the official training and test split, which separates participants. It compares:

1. **Majority baseline:** provides a simple reference point.
2. **Random Forest:** a tree-ensemble model.
3. **RBF Support Vector Machine:** a nonlinear classifier with standardized features.

The model with the highest test-set macro-F1 is saved for demonstration. For a publication-quality experiment, model selection and hyperparameter tuning should be performed using training-only cross-validation, with the test set reserved for one final evaluation.

## Results

Run the experiment locally and use the generated files in `outputs/` to report actual results. Do not add an accuracy claim to a CV or report until you have verified the run output. Accuracy varies by implementation and environment; macro-F1 and per-class results should be reported alongside it.

## Limitations and future work

- Uses pre-engineered features rather than raw, live smartphone sensor streams.
- Does not include a fall label or validate fall detection.
- Does not establish medical or clinical safety.
- Future work could evaluate subject-independent generalization, training-only cross-validation, feature ablation, model calibration, inference latency, and energy use.
- A separate fall-detection study should use an appropriate fall dataset, report missed falls and false alarms, and evaluate participants not used during training.

## Author

**Muhammad Hasnain Maqsood**  
BS Software Engineering, Thal University of Bhakkar, Pakistan

## Dataset citation

Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository. https://doi.org/10.24432/C54G4K

Dataset DOI: `10.24432/C54G4K`
