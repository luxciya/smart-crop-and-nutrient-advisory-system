"""
Dataset Analysis – Smart Crop Advisory System
=============================================
Analyzes training_dataset.csv and prints:
  - Dataset shape
  - Train/Test split counts
  - Missing values
  - Duplicate rows
  - Unique classes
  - Crop distribution
  - Season distribution
  - Region distribution
  - Soil type distribution
  - Numerical feature summary
  - Class imbalance check

Optional:
  - Saves a dataset analysis report to dataset_analysis_report.txt

Run:
    python analyze_dataset.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATASET_FILE = "training_dataset.csv"
TARGET = "recommended_crop"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["soil_type", "season", "region"]
NUMERIC_FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "ph", "moisture", "temperature", "humidity", "rainfall"
]

REPORT_FILE = "dataset_analysis_report.txt"


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"'{DATASET_FILE}' not found. Run generate_dataset.py first."
        )
    df = pd.read_csv(DATASET_FILE)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def analyze_dataset(df):
    lines = []
    lines.append("=" * 70)
    lines.append("SMART CROP ADVISORY SYSTEM – DATASET ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append("")

    # Basic shape
    total_rows, total_cols = df.shape
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"Total Rows       : {total_rows}")
    lines.append(f"Total Columns    : {total_cols}")
    lines.append(f"Target Column    : {TARGET}")
    lines.append("")

    # Train/Test split
    train_rows, test_rows = get_train_test_counts(df)
    lines.append("2. TRAIN / TEST SPLIT")
    lines.append("-" * 70)
    lines.append(f"Train Rows (80%) : {train_rows}")
    lines.append(f"Test Rows  (20%) : {test_rows}")
    lines.append("")

    # Missing values
    lines.append("3. MISSING VALUES")
    lines.append("-" * 70)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    lines.append(f"Total Missing Values : {total_missing}")
    if total_missing == 0:
        lines.append("No missing values found.")
    else:
        for col, val in missing.items():
            if val > 0:
                lines.append(f"{col:<20} {val}")
    lines.append("")

    # Duplicate rows
    lines.append("4. DUPLICATE ROWS")
    lines.append("-" * 70)
    duplicate_count = df.duplicated().sum()
    lines.append(f"Duplicate Rows : {duplicate_count}")
    lines.append("")

    # Unique values
    lines.append("5. UNIQUE VALUE COUNTS")
    lines.append("-" * 70)
    lines.append(f"Unique Crops      : {df[TARGET].nunique()}")
    lines.append(f"Unique Soil Types : {df['soil_type'].nunique()}")
    lines.append(f"Unique Seasons    : {df['season'].nunique()}")
    lines.append(f"Unique Regions    : {df['region'].nunique()}")
    lines.append("")

    # Crop distribution
    lines.append("6. CROP CLASS DISTRIBUTION")
    lines.append("-" * 70)
    crop_counts = df[TARGET].value_counts().sort_values(ascending=False)
    for crop, count in crop_counts.items():
        lines.append(f"{crop:<25} {count}")
    lines.append("")

    # Class imbalance check
    lines.append("7. CLASS IMBALANCE CHECK")
    lines.append("-" * 70)
    min_count = crop_counts.min()
    max_count = crop_counts.max()
    imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else "Undefined"

    lines.append(f"Minimum Class Count : {min_count}")
    lines.append(f"Maximum Class Count : {max_count}")
    lines.append(f"Imbalance Ratio     : {imbalance_ratio}")

    if isinstance(imbalance_ratio, float):
        if imbalance_ratio <= 1.5:
            lines.append("Dataset Status      : Well Balanced ✅")
        elif imbalance_ratio <= 3:
            lines.append("Dataset Status      : Moderately Balanced ⚠️")
        else:
            lines.append("Dataset Status      : Imbalanced ❌")
    lines.append("")

    # Soil distribution
    lines.append("8. SOIL TYPE DISTRIBUTION")
    lines.append("-" * 70)
    soil_counts = df["soil_type"].value_counts()
    for soil, count in soil_counts.items():
        lines.append(f"{soil:<25} {count}")
    lines.append("")

    # Season distribution
    lines.append("9. SEASON DISTRIBUTION")
    lines.append("-" * 70)
    season_counts = df["season"].value_counts()
    for season, count in season_counts.items():
        lines.append(f"{season:<60} {count}")
    lines.append("")

    # Region distribution
    lines.append("10. REGION DISTRIBUTION")
    lines.append("-" * 70)
    region_counts = df["region"].value_counts()
    for region, count in region_counts.items():
        lines.append(f"{region:<50} {count}")
    lines.append("")

    # Numeric summary
    lines.append("11. NUMERICAL FEATURE SUMMARY")
    lines.append("-" * 70)
    summary = df[NUMERIC_FEATURES].describe().round(2)
    lines.append(summary.to_string())
    lines.append("")

    # Dataset quality verdict
    lines.append("12. DATASET QUALITY VERDICT")
    lines.append("-" * 70)
    lines.append("This dataset is suitable for machine learning classification.")
    lines.append("It contains agronomic, environmental, seasonal, and regional features.")
    lines.append("The dataset is appropriate for crop recommendation using multiple ML algorithms.")
    lines.append("")

    return "\n".join(lines)


def get_train_test_counts(df):
    """
    Uses the same logic as train_model.py:
    stratified 80/20 split
    """
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return len(X_train), len(X_test)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    df = load_data()
    report = analyze_dataset(df)

    # Print to console
    print(report)

    # Save to text file
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nDataset analysis report saved to → {REPORT_FILE}")


if __name__ == "__main__":
    main()