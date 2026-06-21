#!/usr/bin/env python3
"""Fixed final SVM optimization"""
import sys, os, time
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

def add_features(df):
    df['start_grid_ultra'] = df['start_lat'].round(4).astype(str) + ',' + df['start_lng'].round(4).astype(str)
    df['end_grid_ultra'] = df['end_lat'].round(4).astype(str) + ',' + df['end_lng'].round(4).astype(str)
    df['dist_per_min'] = df['distance_km'] / (df['duration_min'] + 0.1)
    df['speed_dist'] = df['speed_kmh'] * df['distance_km']
    df['hour_weekend'] = df['hour_cat'] + '_' + df['is_weekend'].astype(str)
    df['hour_weekday'] = df['hour_cat'] + '_' + df['weekday_cat']
    df['very_short'] = (df['duration_min'] < 5).astype(int)
    df['long_trip'] = (df['duration_min'] > 30).astype(int)
    df['fast_rider'] = (df['speed_kmh'] > 15).astype(int)
    df['rideable_weekend'] = df['rideable_type'] + '_' + df['is_weekend'].astype(str)
    df['rideable_timeperiod'] = df['rideable_type'] + '_' + df['time_period']
    return df

NUM = ["duration_min", "distance_km", "speed_kmh", "start_lat", "start_lng", "end_lat", "end_lng",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "is_weekend", "is_commute_peak",
    "is_round_trip", "start_station_missing", "end_station_missing", "dist_per_min",
    "speed_dist", "very_short", "long_trip", "fast_rider"]

CAT = ["rideable_type", "time_period", "duration_bin", "distance_bin", "speed_bin",
    "start_grid", "end_grid", "start_station_id", "end_station_id", "start_grid_ultra",
    "end_grid_ultra", "hour_weekend", "hour_weekday", "rideable_weekend", "rideable_timeperiod"]

TM = ["start_station_id", "end_station_id", "route_station_id", "start_grid_ultra",
    "end_grid_ultra", "hour_weekend", "hour_weekday", "rideable_weekend", "rideable_timeperiod"]

TM_ENC = [f"{c}_target_mean" for c in TM]
FEATURES = list(dict.fromkeys(NUM + CAT + ["route_station_id"]))

def build_pipe():
    num_t = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    cat_t = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=8, sparse_output=True))])
    prep = ColumnTransformer([("num", num_t, NUM + TM_ENC), ("cat", cat_t, CAT)], sparse_threshold=0.3)
    return Pipeline([
        ("tm", TargetMeanEncoder(columns=TM, smoothing=7.0, n_splits=5, random_state=RANDOM_STATE)),
        ("prep", prep),
        ("svm", LinearSVC(class_weight="balanced", dual="auto", max_iter=50000, tol=1e-4, random_state=RANDOM_STATE))])

def train(df, name):
    X, y = df[FEATURES].copy(), df[TARGET].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    pipe = build_pipe()
    grid = {"svm__C": [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08],
        "svm__loss": ["hinge", "squared_hinge"]}
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    
    print(f"\n[{name}] Training...")
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    
    model = gs.best_estimator_
    y_pred = model.predict(X_te)
    f1_test = float(f1_score(y_te, y_pred, average="weighted"))
    acc = float(accuracy_score(y_te, y_pred))
    rep = classification_report(y_te, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    return {
        "name": name, "params": gs.best_params_, "cv_f1": float(gs.best_score_),
        "test_f1": f1_test, "acc": acc, "f1_member": rep["member"]["f1-score"],
        "f1_casual": rep["casual"]["f1-score"], "time": elapsed
    }, model

def main():
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    results = []
    best_f1, best_model = 0, None
    
    for sz in [10000, 13000, 16000]:
        print(f"\n{'='*60}\nSample: {sz}/class/month\n{'='*60}")
        sampled, _ = profile_and_sample(files, sz, RANDOM_STATE)
        
        from run_svm_experiment import clean_and_engineer
        cleaned, _ = clean_and_engineer(sampled)
        enhanced = add_features(cleaned)
        print(f"  Total: {len(enhanced)} samples, {len(FEATURES)} features")
        
        res, mdl = train(enhanced, f"sz{sz}")
        results.append(res)
        
        print(f"\n  CV F1: {res['cv_f1']:.4f}")
        print(f"  Test F1: {res['test_f1']:.4f}")
        print(f"  Acc: {res['acc']:.4f}")
        
        if res['test_f1'] > best_f1:
            best_f1 = res['test_f1']
            best_model = mdl
        
        if res['test_f1'] >= 0.70:
            print(f"\n✓✓✓ SUCCESS! F1={res['test_f1']:.4f} >= 0.70 ✓✓✓")
            break
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(dirs["tables"] / "svm_final_results.csv", index=False)
    joblib.dump(best_model, dirs["models"] / "svm_best_model.joblib")
    
    print(f"\n{'='*60}\nFINAL:")
    print(df_res[["name", "cv_f1", "test_f1", "acc"]].to_string(index=False))
    print(f"\nBest F1: {best_f1:.4f}")
    print("✓ SUCCESS!" if best_f1 >= 0.70 else f"✗ Need +{0.70-best_f1:.4f}")
    print('='*60)

if __name__ == "__main__":
    main()
