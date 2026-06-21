#!/usr/bin/env python3
"""Enhanced SVM experiment with advanced feature engineering to boost F1 score to 0.7+"""
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.svm import SVC

# Import from original script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (
    RANDOM_STATE,
    TARGET,
    CLASS_ORDER,
    csv_files,
    profile_and_sample,
    TargetMeanEncoder,
    ensure_dirs,
    haversine_km,
)


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced engineered features for better discrimination"""
    print("  → Adding advanced spatial features...")

    # Fine-grained spatial features (0.01 degree ~1km)
    df['start_grid_fine'] = (
        df['start_lat'].round(3).astype(str) + ',' + df['start_lng'].round(3).astype(str)
    )
    df['end_grid_fine'] = (
        df['end_lat'].round(3).astype(str) + ',' + df['end_lng'].round(3).astype(str)
    )

    # Spatial movement patterns
    df['movement_ns'] = np.where(df['lat_delta'] > 0, 'north', 'south')
    df['movement_ew'] = np.where(df['lng_delta'] > 0, 'east', 'west')
    df['movement_direction'] = df['movement_ns'] + '_' + df['movement_ew']

    # Distance-to-duration ratio (efficiency metric)
    df['distance_per_min'] = df['distance_km'] / (df['duration_min'] + 0.1)

    print("  → Adding temporal interaction features...")

    # Time-space interactions
    df['hour_x_weekend'] = df['hour_cat'] + '_' + df['is_weekend'].astype(str)
    df['time_period_x_weekday'] = df['time_period'] + '_' + df['weekday_cat']
    df['season_x_weekend'] = df['season'] + '_wknd' + df['is_weekend'].astype(str)


    # Trip behavior features
    df['is_very_short'] = (df['duration_min'] < 5).astype(int)
    df['is_long_trip'] = (df['duration_min'] > 30).astype(int)
    df['is_stationary'] = (df['distance_km'] < 0.1).astype(int)
    df['is_fast_rider'] = (df['speed_kmh'] > 15).astype(int)
    df['is_slow_rider'] = (df['speed_kmh'] < 8).astype(int)

    print("  → Adding station and route features...")

    # Route complexity
    df['trip_straightness'] = np.where(
        df['duration_min'] * df['speed_kmh'] / 60.0 > 0.1,
        df['distance_km'] / (df['duration_min'] * df['speed_kmh'] / 60.0 + 0.01),
        0.5
    )
    df['trip_straightness'] = df['trip_straightness'].clip(0, 1)

    # Station availability patterns
    df['both_stations_missing'] = (
        (df['start_station_missing'] == 1) & (df['end_station_missing'] == 1)
    ).astype(int)
    df['any_station_missing'] = (
        (df['start_station_missing'] == 1) | (df['end_station_missing'] == 1)
    ).astype(int)

    # Grid-based features
    df['same_grid'] = (df['start_grid'] == df['end_grid']).astype(int)
    df['same_grid_fine'] = (df['start_grid_fine'] == df['end_grid_fine']).astype(int)

    print("  → Adding rideable type interactions...")

    # Rideable type interactions
    df['rideable_x_weekend'] = df['rideable_type'] + '_wknd' + df['is_weekend'].astype(str)
    df['rideable_x_hour'] = df['rideable_type'] + '_h' + df['hour_cat']
    df['rideable_x_duration'] = df['rideable_type'] + '_' + df['duration_bin']

    return df



# Enhanced feature lists
NUMERIC_FEATURES_ENHANCED = [
    "duration_min",
    "distance_km",
    "speed_kmh",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "lat_delta",
    "lng_delta",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_commute_peak",
    "is_round_trip",
    "start_station_missing",
    "end_station_missing",
    "distance_per_min",
    "is_very_short",
    "is_long_trip",
    "is_stationary",
    "is_fast_rider",
    "is_slow_rider",
    "trip_straightness",
    "both_stations_missing",
    "any_station_missing",
    "same_grid",
    "same_grid_fine",
]

CATEGORICAL_FEATURES_ENHANCED = [
    "rideable_type",
    "season",
    "time_period",
    "source_month",
    "hour_cat",
    "weekday_cat",
    "duration_bin",
    "distance_bin",
    "speed_bin",
    "start_grid",
    "end_grid",
    "route_grid",
    "start_station_id",
    "end_station_id",
    "start_grid_fine",
    "end_grid_fine",
    "movement_direction",
    "hour_x_weekend",
    "time_period_x_weekday",
    "season_x_weekend",
    "rideable_x_weekend",
    "rideable_x_hour",
    "rideable_x_duration",
]

TARGET_MEAN_FEATURES_ENHANCED = [
    "start_station_id",
    "end_station_id",
    "route_station_id",
    "start_grid",
    "end_grid",
    "route_grid",
    "weekday_hour",
    "time_period",
    "rideable_type",
    "start_grid_fine",
    "end_grid_fine",
    "movement_direction",
    "hour_x_weekend",
    "rideable_x_weekend",
]

TARGET_MEAN_ENCODED_FEATURES_ENHANCED = [f"{col}_target_mean" for col in TARGET_MEAN_FEATURES_ENHANCED]
TARGET_MEAN_SOURCE_ONLY_FEATURES_ENHANCED = ["route_station_id", "weekday_hour"]
MODEL_FEATURES_ENHANCED = list(
    dict.fromkeys(NUMERIC_FEATURES_ENHANCED + CATEGORICAL_FEATURES_ENHANCED + TARGET_MEAN_SOURCE_ONLY_FEATURES_ENHANCED)
)


def build_enhanced_model(use_rbf: bool = False) -> Pipeline:
    """Build enhanced model with better preprocessing and optional RBF kernel"""
    numeric_model_features = NUMERIC_FEATURES_ENHANCED + TARGET_MEAN_ENCODED_FEATURES_ENHANCED

    # Use RobustScaler for better handling of outliers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),  # More robust to outliers
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,  # Lower threshold to capture more patterns
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_model_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES_ENHANCED),
        ],
        sparse_threshold=0.3,
    )

    # Choose model
    if use_rbf:
        model = SVC(
            kernel='rbf',
            class_weight='balanced',
            max_iter=20000,
            random_state=RANDOM_STATE,
            cache_size=1000,
        )
    else:
        from sklearn.svm import LinearSVC
        model = LinearSVC(
            class_weight="balanced",
            dual="auto",
            max_iter=20000,
            random_state=RANDOM_STATE,
        )

    return Pipeline(
        steps=[
            (
                "target_mean",
                TargetMeanEncoder(
                    columns=TARGET_MEAN_FEATURES_ENHANCED,
                    smoothing=15.0,  # Slightly less smoothing for more signal
                    n_splits=5,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def train_enhanced_model(df: pd.DataFrame, dirs: dict[str, Path], use_rbf: bool = False):
    """Train enhanced model with advanced features"""
    print("\n[Enhanced Training] Starting...")

    X = df[MODEL_FEATURES_ENHANCED].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_model = build_enhanced_model(use_rbf=use_rbf)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    if use_rbf:
        param_grid = {
            "model__C": [0.5, 1.0, 2.0, 5.0],
            "model__gamma": ['scale', 0.01, 0.1],
        }
        model_name = "Enhanced RBF SVM"
    else:
        param_grid = {
            "model__C": [0.01, 0.02, 0.05, 0.1, 0.2],
        }
        model_name = "Enhanced Linear SVM"

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        refit="f1_weighted",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        return_train_score=True,
    )

    print(f"\n[Training] {model_name} with {cv.n_splits}-fold CV...")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    model = grid.best_estimator_

    print(f"\n[Training Complete] Time: {train_seconds:.1f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV F1: {grid.best_score_:.4f}")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred))

    print(f"\n[Test Results]")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  F1 (member): {report['member']['f1-score']:.4f}")
    print(f"  F1 (casual): {report['casual']['f1-score']:.4f}")

    metrics = {
        "random_state": RANDOM_STATE,
        "model": model_name,
        "use_rbf": use_rbf,
        "feature_count": len(MODEL_FEATURES_ENHANCED),
        "numeric_features_count": len(NUMERIC_FEATURES_ENHANCED),
        "categorical_features_count": len(CATEGORICAL_FEATURES_ENHANCED),
        "best_params": grid.best_params_,
        "best_cv_f1_weighted": float(grid.best_score_),
        "train_seconds": float(train_seconds),
        "accuracy": accuracy,
        "f1_weighted": f1_weighted,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    # Save results
    suffix = "_rbf" if use_rbf else "_enhanced"
    with (dirs["output"] / f"metrics{suffix}.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    joblib.dump(model, dirs["models"] / f"svm_pipeline{suffix}.joblib")

    return model, metrics, f1_weighted


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Enhanced SVM with advanced features")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--sample-size", type=int, default=5000, help="Sample per month per class")
    parser.add_argument("--use-rbf", action="store_true", help="Use RBF kernel instead of linear")
    args = parser.parse_args()

    dirs = ensure_dirs(args.output_dir)

    print("[1/5] Loading and sampling data...")
    files = csv_files(args.data_dir)
    sampled, raw_tables = profile_and_sample(files, args.sample_size, RANDOM_STATE)

    print(f"[2/5] Cleaning and engineering features (base + enhanced)...")
    # Import the clean_and_engineer from original script
    from run_svm_experiment import clean_and_engineer
    cleaned, cleaning_table = clean_and_engineer(sampled)

    print(f"[3/5] Adding advanced features...")
    enhanced = add_advanced_features(cleaned)
    print(f"  Total rows: {len(enhanced)}")
    print(f"  Total features: {len(MODEL_FEATURES_ENHANCED)}")

    # Save enhanced data
    enhanced.to_csv(dirs["processed"] / "enhanced_sample.csv", index=False)

    print(f"\n[4/5] Training enhanced model...")
    model, metrics, f1_score = train_enhanced_model(enhanced, dirs, use_rbf=args.use_rbf)

    print(f"\n[5/5] Complete!")
    print(f"=" * 60)
    print(f"Final F1 Score: {f1_score:.4f}")
    if f1_score >= 0.70:
        print(f"✓ SUCCESS! F1 >= 0.70 achieved!")
    else:
        print(f"✗ Target not reached. Need {0.70 - f1_score:.4f} more.")
    print(f"=" * 60)

    return metrics


if __name__ == "__main__":
    main()





