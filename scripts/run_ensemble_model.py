#!/usr/bin/env python3
"""Ensemble models (Random Forest, XGBoost) to achieve F1 >= 0.7"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

# Import from original
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (
    RANDOM_STATE,
    TARGET,
    CLASS_ORDER,
    csv_files,
    profile_and_sample,
    ensure_dirs,
)
from run_svm_enhanced import (
    add_advanced_features,
    NUMERIC_FEATURES_ENHANCED,
    CATEGORICAL_FEATURES_ENHANCED,
    TARGET_MEAN_SOURCE_ONLY_FEATURES_ENHANCED,
    MODEL_FEATURES_ENHANCED,
)


def prepare_features_for_tree_models(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for tree-based models"""
    df_encoded = df.copy()

    # Label encode categorical features
    from sklearn.preprocessing import LabelEncoder

    for col in CATEGORICAL_FEATURES_ENHANCED:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    for col in TARGET_MEAN_SOURCE_ONLY_FEATURES_ENHANCED:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded


def train_random_forest(df: pd.DataFrame, dirs: dict[str, Path]):
    """Train Random Forest classifier"""
    print("\n[Random Forest Training] Starting...")

    df_encoded = prepare_features_for_tree_models(df)
    X = df_encoded[MODEL_FEATURES_ENHANCED].copy()
    y = df_encoded[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Random Forest with grid search
    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced',
    )

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [20, 30, None],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'max_features': ['sqrt', 'log2'],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        rf,
        param_grid,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    print(f"Training Random Forest with {cv.n_splits}-fold CV...")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    f1 = float(f1_score(y_test, y_pred, average='weighted'))
    acc = float(accuracy_score(y_test, y_pred))

    print(f"\n[Random Forest Results]")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1: {grid.best_score_:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1: {f1:.4f}")
    print(f"  Training time: {train_time:.1f}s")

    # Save model
    joblib.dump(model, dirs["models"] / "random_forest_model.joblib")

    return model, f1, grid.best_score_


def train_xgboost(df: pd.DataFrame, dirs: dict[str, Path]):
    """Train XGBoost classifier"""
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed, skipping...")
        return None, 0.0, 0.0

    print("\n[XGBoost Training] Starting...")

    df_encoded = prepare_features_for_tree_models(df)
    X = df_encoded[MODEL_FEATURES_ENHANCED].copy()
    y = df_encoded[TARGET].copy()

    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE
    )

    # Calculate scale_pos_weight for imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb_model = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
    )

    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        xgb_model,
        param_grid,
        scoring='f1_weighted',
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    print(f"Training XGBoost with {cv.n_splits}-fold CV...")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start

    model = grid.best_estimator_
    y_pred = model.predict(X_test)

    # Decode predictions for metrics
    y_test_decoded = le_target.inverse_transform(y_test)
    y_pred_decoded = le_target.inverse_transform(y_pred)

    f1 = float(f1_score(y_test_decoded, y_pred_decoded, average='weighted'))
    acc = float(accuracy_score(y_test_decoded, y_pred_decoded))

    print(f"\n[XGBoost Results]")
    print(f"  Best params: {grid.best_params_}")
    print(f"  Best CV F1: {grid.best_score_:.4f}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1: {f1:.4f}")
    print(f"  Training time: {train_time:.1f}s")

    # Save model
    joblib.dump((model, le_target), dirs["models"] / "xgboost_model.joblib")

    return model, f1, grid.best_score_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--model", choices=['rf', 'xgb', 'both'], default='both')
    args = parser.parse_args()

    dirs = ensure_dirs(args.output_dir)

    print("[1/4] Loading data...")
    files = csv_files(args.data_dir)
    sampled, _ = profile_and_sample(files, args.sample_size, RANDOM_STATE)

    print("[2/4] Feature engineering...")
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_advanced_features(cleaned)
    print(f"  Total samples: {len(enhanced)}")

    results = {}

    if args.model in ['rf', 'both']:
        print("\n[3/4] Training Random Forest...")
        _, rf_f1, rf_cv = train_random_forest(enhanced, dirs)
        results['random_forest'] = {'test_f1': rf_f1, 'cv_f1': rf_cv}

    if args.model in ['xgb', 'both']:
        print("\n[4/4] Training XGBoost...")
        _, xgb_f1, xgb_cv = train_xgboost(enhanced, dirs)
        results['xgboost'] = {'test_f1': xgb_f1, 'cv_f1': xgb_cv}

    print("\n" + "="*60)
    print("FINAL RESULTS:")
    for model_name, scores in results.items():
        print(f"{model_name:20s} Test F1: {scores['test_f1']:.4f}  CV F1: {scores['cv_f1']:.4f}")
        if scores['test_f1'] >= 0.70:
            print(f"  ✓ {model_name} achieved F1 >= 0.70!")
    print("="*60)

    # Save results
    with (dirs["output"] / "ensemble_results.json").open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
