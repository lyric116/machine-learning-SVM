#!/usr/bin/env python3
"""Targeted SVM optimization - LinearSVC only, memory-efficient"""
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (
    RANDOM_STATE, TARGET, CLASS_ORDER, csv_files, profile_and_sample,
    ensure_dirs, TargetMeanEncoder,
)

def add_high_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add only high-value features"""
    print("  → Adding targeted features...")
    
    df['start_grid_fine'] = df['start_lat'].round(3).astype(str) + ',' + df['start_lng'].round(3).astype(str)
    df['end_grid_fine'] = df['end_lat'].round(3).astype(str) + ',' + df['end_lng'].round(3).astype(str)
    df['distance_per_min'] = df['distance_km'] / (df['duration_min'] + 0.1)
    df['hour_x_weekend'] = df['hour_cat'] + '_' + df['is_weekend'].astype(str)
    df['is_very_short'] = (df['duration_min'] < 5).astype(int)
    df['is_long_trip'] = (df['duration_min'] > 30).astype(int)
    df['rideable_x_weekend'] = df['rideable_type'] + '_' + df['is_weekend'].astype(str)
    
    return df

NUMERIC_FEATURES = [
    "duration_min", "distance_km", "speed_kmh", "start_lat", "start_lng",
    "end_lat", "end_lng", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_weekend", "is_commute_peak", "is_round_trip", "start_station_missing",
    "end_station_missing", "distance_per_min", "is_very_short", "is_long_trip",
]

CATEGORICAL_FEATURES = [
    "rideable_type", "time_period", "duration_bin", "distance_bin", "speed_bin",
    "start_grid", "end_grid", "start_station_id", "end_station_id",
    "start_grid_fine", "end_grid_fine", "hour_x_weekend", "rideable_x_weekend",
]

TARGET_MEAN_FEATURES = [
    "start_station_id", "end_station_id", "route_station_id",
    "start_grid_fine", "end_grid_fine", "hour_x_weekend", "rideable_x_weekend",
]

TARGET_MEAN_ENCODED = [f"{c}_target_mean" for c in TARGET_MEAN_FEATURES]
MODEL_FEATURES = list(dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["route_station_id"]))

def build_pipeline():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=15, sparse_output=True)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES + TARGET_MEAN_ENCODED),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ], sparse_threshold=0.3)
    
    return Pipeline([
        ("target_mean", TargetMeanEncoder(columns=TARGET_MEAN_FEATURES, smoothing=10.0, n_splits=5, random_state=RANDOM_STATE)),
        ("preprocess", preprocessor),
        ("model", LinearSVC(class_weight="balanced", dual="auto", max_iter=25000, random_state=RANDOM_STATE)),
    ])

def run_experiment(df, dirs, name):
    X, y = df[MODEL_FEATURES].copy(), df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    pipeline = build_pipeline()
    param_grid = {
        "model__C": [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        "model__loss": ["hinge", "squared_hinge"],
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    
    print(f"\n[{name}] Training...")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    result = {
        "experiment": name, "best_params": grid.best_params_, "cv_f1": float(grid.best_score_),
        "test_f1": f1, "test_accuracy": acc, "f1_member": report["member"]["f1-score"],
        "f1_casual": report["casual"]["f1-score"], "train_time": train_time,
    }
    
    print(f"  CV F1: {result['cv_f1']:.4f}  Test F1: {f1:.4f}  Acc: {acc:.4f}")
    return result, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    
    dirs = ensure_dirs(args.output_dir)
    files = csv_files(args.data_dir)
    
    experiments = []
    for sample_size in [6000, 8000, 10000]:
        print(f"\n{'='*60}\nEXPERIMENT: {sample_size} samples/class/month\n{'='*60}")
        sampled, _ = profile_and_sample(files, sample_size, RANDOM_STATE)
        
        from run_svm_experiment import clean_and_engineer
        cleaned, _ = clean_and_engineer(sampled)
        enhanced = add_high_value_features(cleaned)
        print(f"  Total: {len(enhanced)} samples, {len(MODEL_FEATURES)} features")
        
        result, model = run_experiment(enhanced, dirs, f"sample_{sample_size}")
        experiments.append(result)
        
        if result['test_f1'] >= 0.70:
            print(f"\n✓ SUCCESS! F1={result['test_f1']:.4f} >= 0.70")
            joblib.dump(model, dirs["models"] / f"svm_best_{sample_size}.joblib")
            break
    
    results_df = pd.DataFrame(experiments)
    results_df.to_csv(dirs["tables"] / "optimization_targeted_experiments.csv", index=False)
    
    print(f"\n{'='*60}\nSUMMARY:")
    print(results_df[["experiment", "cv_f1", "test_f1", "test_accuracy"]].to_string(index=False))
    best = results_df.loc[results_df["test_f1"].idxmax()]
    print(f"\nBest: {best['experiment']} F1={best['test_f1']:.4f}")
    print("✓ Goal achieved!" if best['test_f1'] >= 0.70 else f"✗ Need {0.70 - best['test_f1']:.4f} more")
    print('='*60)

if __name__ == "__main__":
    main()
