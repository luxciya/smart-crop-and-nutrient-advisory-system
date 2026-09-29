"""
Model Training – Smart Crop Advisory System
============================================
Trains FIVE models on training_dataset.csv:
  1. Random Forest Classifier
  2. Logistic Regression
  3. Support Vector Machine (SVM)
  4. XGBoost Classifier
  5. CatBoost Classifier

Evaluates each model using:
  - Accuracy
  - Precision
  - Recall
  - F1-score

Saves:
  models/random_forest_model.pkl
  models/logistic_regression_model.pkl
  models/svm_model.pkl
  models/xgboost_model.pkl
  models/catboost_model.pkl
  models/label_encoder.pkl
  models/feature_encoder.pkl
  models/scaler.pkl
"""

import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DATASET_FILE = "training_dataset.csv"
MODEL_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42

CATEGORICAL_FEATURES = ["soil_type", "season", "region"]
NUMERIC_FEATURES = [
    "nitrogen", "phosphorus", "potassium",
    "ph", "moisture", "temperature", "humidity", "rainfall"
]
TARGET = "recommended_crop"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ──────────────────────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(
            f"'{DATASET_FILE}' not found. Run generate_dataset.py first."
        )
    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded {len(df)} rows, {df[TARGET].nunique()} unique crops.")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESS
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(df):
    # Encode categorical input features
    feature_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df[CATEGORICAL_FEATURES] = feature_encoder.fit_transform(df[CATEGORICAL_FEATURES])

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df[TARGET])

    X = df[ALL_FEATURES].values
    y = y_encoded

    print(f"Features : {ALL_FEATURES}")
    print(f"Classes  : {list(label_encoder.classes_)}\n")

    return X, y, feature_encoder, label_encoder


# ──────────────────────────────────────────────────────────────────────────────
# 3. COMMON METRICS FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    return acc, precision, recall, f1


def print_metrics(acc, precision, recall, f1):
    print(f"Accuracy  : {acc * 100:.2f}%")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1 Score  : {f1 * 100:.2f}%\n")


# ──────────────────────────────────────────────────────────────────────────────
# 4. TRAIN RANDOM FOREST
# ──────────────────────────────────────────────────────────────────────────────
def train_random_forest(X_train, X_test, y_train, y_test, label_encoder):
    print("=" * 60)
    print("  MODEL 1 : Random Forest Classifier")
    print("=" * 60)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print_metrics(acc, precision, recall, f1)

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    # Feature importances
    importances = sorted(
        zip(ALL_FEATURES, rf.feature_importances_),
        key=lambda x: x[1], reverse=True
    )
    print("Top Feature Importances:")
    for feat, imp in importances:
        print(f"  {feat:<20} {imp:.4f}")
    print()

    return rf, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# 5. TRAIN LOGISTIC REGRESSION (WITH SCALING)
# ──────────────────────────────────────────────────────────────────────────────
def train_logistic_regression(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder):
    print("=" * 60)
    print("  MODEL 2 : Logistic Regression (Scaled)")
    print("=" * 60)

    log_reg = LogisticRegression(
        max_iter=5000,
        solver='lbfgs',
        multi_class='multinomial',
        random_state=RANDOM_STATE
    )
    log_reg.fit(X_train_scaled, y_train)

    y_pred = log_reg.predict(X_test_scaled)
    acc, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print_metrics(acc, precision, recall, f1)

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    return log_reg, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# 6. TRAIN SUPPORT VECTOR MACHINE (SVM) (WITH SCALING)
# ──────────────────────────────────────────────────────────────────────────────
def train_svm(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder):
    print("=" * 60)
    print("  MODEL 3 : Support Vector Machine (SVM) (Scaled)")
    print("=" * 60)

    svm_model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,
        random_state=RANDOM_STATE
    )
    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)
    acc, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print_metrics(acc, precision, recall, f1)

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    return svm_model, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# 7. TRAIN XGBOOST CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────
def train_xgboost(X_train, X_test, y_train, y_test, label_encoder):
    print("=" * 60)
    print("  MODEL 4 : XGBoost Classifier")
    print("=" * 60)

    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.08,
        objective='multi:softmax',
        num_class=len(label_encoder.classes_),
        random_state=RANDOM_STATE,
        eval_metric='mlogloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    acc, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print_metrics(acc, precision, recall, f1)

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    return xgb_model, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# 8. TRAIN CATBOOST CLASSIFIER
# ──────────────────────────────────────────────────────────────────────────────
def train_catboost(X_train, X_test, y_train, y_test, label_encoder):
    print("=" * 60)
    print("  MODEL 5 : CatBoost Classifier")
    print("=" * 60)

    cat_model = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.08,
        loss_function='MultiClass',
        verbose=0,
        random_seed=RANDOM_STATE
    )
    cat_model.fit(X_train, y_train)

    y_pred = cat_model.predict(X_test)
    y_pred = y_pred.flatten().astype(int)

    acc, precision, recall, f1 = evaluate_model(y_test, y_pred)

    print_metrics(acc, precision, recall, f1)

    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    ))

    return cat_model, acc, precision, recall, f1


# ──────────────────────────────────────────────────────────────────────────────
# 9. SAVE MODELS
# ──────────────────────────────────────────────────────────────────────────────
def save_model(obj, filename):
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load & preprocess
    df = load_data()
    X, y, feature_encoder, label_encoder = preprocess(df)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train samples : {len(X_train)}")
    print(f"Test  samples : {len(X_test)}\n")

    # ──────────────────────────────────────────────────────────────────────────
    # SCALE ONLY FOR LOGISTIC REGRESSION + SVM
    # ──────────────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train all 5 models
    rf_model, rf_acc, rf_precision, rf_recall, rf_f1 = train_random_forest(
        X_train, X_test, y_train, y_test, label_encoder
    )

    log_model, log_acc, log_precision, log_recall, log_f1 = train_logistic_regression(
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder
    )

    svm_model, svm_acc, svm_precision, svm_recall, svm_f1 = train_svm(
        X_train_scaled, X_test_scaled, y_train, y_test, label_encoder
    )

    xgb_model, xgb_acc, xgb_precision, xgb_recall, xgb_f1 = train_xgboost(
        X_train, X_test, y_train, y_test, label_encoder
    )

    cat_model, cat_acc, cat_precision, cat_recall, cat_f1 = train_catboost(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Save all artefacts
    print("=" * 60)
    print("  SAVING MODELS")
    print("=" * 60)
    save_model(rf_model, "random_forest_model.pkl")
    save_model(log_model, "logistic_regression_model.pkl")
    save_model(svm_model, "svm_model.pkl")
    save_model(xgb_model, "xgboost_model.pkl")
    save_model(cat_model, "catboost_model.pkl")
    save_model(label_encoder, "label_encoder.pkl")
    save_model(feature_encoder, "feature_encoder.pkl")
    save_model(scaler, "scaler.pkl")

    print("\nDone! All 5 models + encoders + scaler saved in the 'models/' folder.")
    print("Load them with: pickle.load(open('models/random_forest_model.pkl', 'rb'))")

    # Save model performance report
    accuracy_file = "model_accuracy.txt"

    with open(accuracy_file, "w", encoding="utf-8") as f:
        f.write("MODEL PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("1. Random Forest Classifier\n")
        f.write(f"Accuracy  : {rf_acc * 100:.2f}%\n")
        f.write(f"Precision : {rf_precision * 100:.2f}%\n")
        f.write(f"Recall    : {rf_recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {rf_f1 * 100:.2f}%\n\n")

        f.write("2. Logistic Regression\n")
        f.write(f"Accuracy  : {log_acc * 100:.2f}%\n")
        f.write(f"Precision : {log_precision * 100:.2f}%\n")
        f.write(f"Recall    : {log_recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {log_f1 * 100:.2f}%\n\n")

        f.write("3. Support Vector Machine (SVM)\n")
        f.write(f"Accuracy  : {svm_acc * 100:.2f}%\n")
        f.write(f"Precision : {svm_precision * 100:.2f}%\n")
        f.write(f"Recall    : {svm_recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {svm_f1 * 100:.2f}%\n\n")

        f.write("4. XGBoost Classifier\n")
        f.write(f"Accuracy  : {xgb_acc * 100:.2f}%\n")
        f.write(f"Precision : {xgb_precision * 100:.2f}%\n")
        f.write(f"Recall    : {xgb_recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {xgb_f1 * 100:.2f}%\n\n")

        f.write("5. CatBoost Classifier\n")
        f.write(f"Accuracy  : {cat_acc * 100:.2f}%\n")
        f.write(f"Precision : {cat_precision * 100:.2f}%\n")
        f.write(f"Recall    : {cat_recall * 100:.2f}%\n")
        f.write(f"F1 Score  : {cat_f1 * 100:.2f}%\n\n")

    print(f"\nPerformance report saved to → {accuracy_file}")


if __name__ == "__main__":
    main()