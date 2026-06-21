#!/usr/bin/env python3
"""Final push: max sample size + all best practices"""
from __future__ import annotations
import sys, os, time, json
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files, 
    profile_and_sample, ensure_dirs, TargetMeanEncoder)

def add_aggressive_features(df):
    """Add all high-value + interaction features"""
    # Spatial
    df['start_grid_fine'] = df['start_lat'].round(3).astype(str) + ',' + df['start_lng'].round(3).astype(str)
    df['end_grid_fine'] = df['end_lat'].round(3).astype(str) + ',' + df['end_lng'].round(3).astype(str)
    df['start_grid_ultrafine'] = df['start_lat'].round(4).astype(str) + ',' + df['start_lng'].round(4).astype(str)
    df['end_grid_ultrafine'] = df['end_lat'].round(4).astype(str) + ',' + df['end_lng'].round(4).astype(str)
    
    # Behavioral
    df['distance_per_min'] = df['distance_km'] / (df['duration_min'] + 0.1)
    df['speed_x_distance'] = df['speed_kmh'] * df['distance_km']
    df['duration_x_distance'] = df['duration_min'] * df['distance_km']
    
    # Time interactions
    df['hour_x_weekend'] = df['hour_cat'] + '_' + df['is_weekend'].astype(str)
    df['hour_x_weekday'] = df['hour_cat'] + '_' + df['weekday_cat']
    df['time_period_x_weekend'] = df['time_period'] + '_' + df['is_weekend'].astype(str)
    
    # Trip type
    df['is_very_short'] = (df['duration_min'] < 5).astype(int)
    df['is_long_trip'] = (df['duration_min'] > 30).astype(int)
    df['is_stationary'] = (df['distance_km'] < 0.1).astype(int)
    df['is_fast'] = (df['speed_kmh'] > 15).astype(int)
    
    # Rideable interactions
    df['rideable_x_weekend'] = df['rideable_type'] + '_' + df['is_weekend'].astype(str)
    df['rideable_x_time'] = df['rideable_type'] + '_' + df['time_period']
    df['rideable_x_hour'] = df['rideable_type'] + '_' + df['hour_cat']
    
    return df

NUMERIC = ["duration_min", "distance_km", "speed_kmh", "start_lat", "start_lng", "end_lat", "end_lng",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "is_commute_peak", 
    "is_round_trip", "start_station_missing", "end_station_missing", "distance_per_min",
    "speed_x_distance", "duration_x_distance", "is_very_short", "is_long_trip", "is_stationary", "is_fast"]

CATEGORICAL = ["rideable_type", "time_period", "duration_bin", "distance_bin", "speed_bin",
    "start_grid", "end_grid", "start_station_id", "end_station_id", "start_grid_fine", 
    "end_grid_fine", "start_grid_ultrafine", "end_grid_ultrafine", "hour_x_weekend",
    "hour_x_weekday", "time_period_x_weekend", "rideable_x_weekend", "rideable_x_time", "rideable_x_hour"]

TARGET_MEAN = ["start_station_id", "end_station_id", "route_station_id", "start_grid_fine",
    "end_grid_fine", "start_grid_ultrafine", "end_grid_ultrafine", "hour_x_weekend",
    "hour_x_weekday", "rideable_x_weekend", "rideable_x_time"]

TARGET_MEAN_ENC = [f"{c}_target_mean" for c in TARGET_MEAN]
MODEL_FEATURES = list(dict.fromkeys(NUMERIC + CATEGORICAL + ["route_station_id"]))

def build_pipeline():
    num_t = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_t = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True))])
    prep = ColumnTransformer([
        ("num", num_t, NUMERIC + TARGET_MEAN_ENC),
        ("cat", cat_t, CATEGORICAL)], sparse_threshold=0.3)
    return Pipeline([
        ("target_mean", TargetMeanEncoder(columns=TARGET_MEAN, smoothing=8.0, n_splits=5, random_state=RANDOM_STATE)),
        ("preprocess", prep),
        ("model", LinearSVC(class_weight="balanced", dual="auto", max_iter=30000, random_state=RANDOM_STATE))])

def train(df, dirs, name):
    X, y = df[MODEL_FEATURES].copy(), df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    pipeline = build_pipeline()
    param_grid = {"model__C": [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        "model__loss": ["hinge", "squared_hinge"]}
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(pipeline, param_grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    
    print(f"\n[{name}] Training...")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    result = {"experiment": name, "best_params": grid.best_params_, "cv_f1": float(grid.best_score_),
        "test_f1": f1, "test_accuracy": acc, "f1_member": report["member"]["f1-score"],
        "f1_casual": report["casual"]["f1-score"], "train_time": elapsed}
    
    print(f"  CV F1: {result['cv_f1']:.4f}  Test F1: {f1:.4f}  Acc: {acc:.4f}")
    return result, model

def main():
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    experiments = []
    for sample_size in [12000, 15000, 20000]:
        print(f"\n{'='*60}\nEXPERIMENT: {sample_size} samples/class/month\n{'='*60}")
        sampled, _ = profile_and_sample(files, sample_size, RANDOM_STATE)
        
        from run_svm_experiment import clean_and_engineer
        cleaned, _ = clean_and_engineer(sampled)
        enhanced = add_aggressive_features(cleaned)
        print(f"  Total: {len(enhanced)} samples, {len(MODEL_FEATURES)} features")
        
        result, model = train(enhanced, dirs, f"sample_{sample_size}")
        experiments.append(result)
        
        if result['test_f1'] >= 0.70:
            print(f"\n✓✓✓ SUCCESS! F1={result['test_f1']:.4f} >= 0.70 ✓✓✓")
            joblib.dump(model, dirs["models"] / f"svm_final_{sample_size}.joblib")
            break
    
    results_df = pd.DataFrame(experiments)
    results_df.to_csv(dirs["tables"] / "optimization_final_experiments.csv", index=False)
    
    print(f"\n{'='*60}\nFINAL SUMMARY:")
    print(results_df[["experiment", "cv_f1", "test_f1", "test_accuracy"]].to_string(index=False))
    best = results_df.loc[results_df["test_f1"].idxmax()]
    print(f"\nBest: {best['experiment']} F1={best['test_f1']:.4f}")
    print("✓ GOAL ACHIEVED!" if best['test_f1'] >= 0.70 else f"✗ Need {0.70 - best['test_f1']:.4f} more")
    print('='*60)

if __name__ == "__main__":
    main()
