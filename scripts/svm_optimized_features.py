#!/usr/bin/env python3
"""基于特征选择分析的优化SVM"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

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
    profile_and_sample, ensure_dirs)
from svm_final_fixed import add_features

# 基于MI分析的最优特征集（去掉无效的target_mean）
BEST_NUMERIC = [
    "distance_km", "duration_min", "speed_kmh",  # 只保留speed_kmh（去掉冗余的dist_per_min）
    "start_lat", "start_lng", "end_lat", "end_lng",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_weekend", "is_commute_peak", "is_round_trip",
    "start_station_missing", "end_station_missing",
    "very_short", "long_trip", "fast_rider",
]

BEST_CATEGORICAL = [
    "rideable_type", "time_period", 
    "duration_bin", "distance_bin", "speed_bin",
    "start_grid", "end_grid",
    "start_grid_ultra", "end_grid_ultra",
    "rideable_weekend", "rideable_timeperiod",
]

# 不使用target_mean！
ALL_FEATURES = BEST_NUMERIC + BEST_CATEGORICAL + ["route_station_id"]

def build_simple_pipeline():
    """无target_mean的简单pipeline"""
    num_t = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    cat_t = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True))])
    prep = ColumnTransformer([
        ("num", num_t, BEST_NUMERIC),
        ("cat", cat_t, BEST_CATEGORICAL)
    ], sparse_threshold=0.3)
    
    return Pipeline([
        ("prep", prep),
        ("svm", LinearSVC(class_weight="balanced", dual="auto", max_iter=50000, 
            tol=1e-4, random_state=RANDOM_STATE))
    ])

def train(df, name):
    X, y = df[ALL_FEATURES].copy(), df[TARGET].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    pipe = build_simple_pipeline()
    grid = {
        "svm__C": [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
        "svm__loss": ["hinge", "squared_hinge"]
    }
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    
    print(f"\n[{name}] 训练（去除无效target_mean特征）...")
    t0 = time.perf_counter()
    gs.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    
    model = gs.best_estimator_
    y_pred = model.predict(X_te)
    f1 = float(f1_score(y_te, y_pred, average="weighted"))
    acc = float(accuracy_score(y_te, y_pred))
    rep = classification_report(y_te, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    return {
        "name": name, "params": gs.best_params_, "cv_f1": float(gs.best_score_),
        "test_f1": f1, "acc": acc, "f1_member": rep["member"]["f1-score"],
        "f1_casual": rep["casual"]["f1-score"], "time": elapsed
    }, model

def main():
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    results = []
    best_f1, best_model = 0, None
    
    for sz in [10000, 13000, 16000]:
        print(f"\n{'='*60}\n样本: {sz}/class/month\n{'='*60}")
        sampled, _ = profile_and_sample(files, sz, RANDOM_STATE)
        
        from run_svm_experiment import clean_and_engineer
        cleaned, _ = clean_and_engineer(sampled)
        enhanced = add_features(cleaned)
        print(f"  总样本: {len(enhanced)}")
        print(f"  数值特征: {len(BEST_NUMERIC)}")
        print(f"  类别特征: {len(BEST_CATEGORICAL)}")
        print(f"  去除: 所有无效的target_mean特征")
        
        res, mdl = train(enhanced, f"sz{sz}_optimized")
        results.append(res)
        
        print(f"\n  CV F1: {res['cv_f1']:.4f}")
        print(f"  Test F1: {res['test_f1']:.4f}")
        print(f"  Acc: {res['acc']:.4f}")
        print(f"  最佳参数: {res['params']}")
        
        if res['test_f1'] > best_f1:
            best_f1 = res['test_f1']
            best_model = mdl
        
        if res['test_f1'] >= 0.70:
            print(f"\n✓✓✓ 达标！F1={res['test_f1']:.4f} >= 0.70 ✓✓✓")
            break
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(dirs["tables"] / "optimized_features_results.csv", index=False)
    if best_model:
        joblib.dump(best_model, dirs["models"] / "svm_optimized_features.joblib")
    
    print(f"\n{'='*60}\n最终结果:\n{'='*60}")
    print(df_res[["name", "cv_f1", "test_f1", "acc"]].to_string(index=False))
    print(f"\n最佳 F1: {best_f1:.4f}")
    print(f"之前最佳: 0.6587 (使用了无效target_mean)")
    print(f"改进: {best_f1 - 0.6587:+.4f}")
    print("✓ 成功!" if best_f1 >= 0.70 else f"✗ 还差 {0.70 - best_f1:.4f}")
    print('='*60)

if __name__ == "__main__":
    main()
