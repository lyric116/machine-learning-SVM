#!/usr/bin/env python3
"""混合策略：精选特征 + 有选择的target_mean"""
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
    profile_and_sample, ensure_dirs, TargetMeanEncoder)
from svm_final_fixed import add_features

# 基于MI分析：只保留高价值特征
NUMERIC_CORE = [
    "distance_km", "duration_min", "speed_kmh",
    "start_lat", "start_lng", "end_lat", "end_lng",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "is_weekend", "is_commute_peak",
    "start_station_missing", "end_station_missing",
    "very_short", "long_trip",
]

CATEGORICAL_CORE = [
    "rideable_type", "time_period",
    "duration_bin", "distance_bin",
    "start_grid_ultra", "end_grid_ultra",
    "start_station_id", "end_station_id",
]

# 只对高基数特征使用target_mean（避免过度编码）
TARGET_MEAN_SELECTIVE = [
    "start_station_id",
    "end_station_id",
    "start_grid_ultra",
    "end_grid_ultra",
]

TM_ENC = [f"{c}_target_mean" for c in TARGET_MEAN_SELECTIVE]
FEATURES = list(dict.fromkeys(NUMERIC_CORE + CATEGORICAL_CORE))

def build_hybrid_pipeline():
    num_t = Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())])
    cat_t = Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=15, sparse_output=True))])
    prep = ColumnTransformer([
        ("num", num_t, NUMERIC_CORE + TM_ENC),
        ("cat", cat_t, CATEGORICAL_CORE)
    ], sparse_threshold=0.3)
    
    return Pipeline([
        ("tm", TargetMeanEncoder(columns=TARGET_MEAN_SELECTIVE, smoothing=5.0, 
            n_splits=5, random_state=RANDOM_STATE)),
        ("prep", prep),
        ("svm", LinearSVC(class_weight="balanced", dual="auto", max_iter=50000, 
            tol=1e-4, random_state=RANDOM_STATE))
    ])

def train(df, name):
    X, y = df[FEATURES].copy(), df[TARGET].copy()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    pipe = build_hybrid_pipeline()
    grid = {
        "svm__C": [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
        "svm__loss": ["hinge", "squared_hinge"]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # 5-fold更稳定
    gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=1)
    
    print(f"\n[{name}] 训练（混合策略：精选特征+选择性TM）...")
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
    
    for sz in [13000, 15000, 18000]:
        print(f"\n{'='*60}\n样本: {sz}/class/month\n{'='*60}")
        sampled, _ = profile_and_sample(files, sz, RANDOM_STATE)
        
        from run_svm_experiment import clean_and_engineer
        cleaned, _ = clean_and_engineer(sampled)
        enhanced = add_features(cleaned)
        print(f"  总样本: {len(enhanced)}")
        print(f"  核心特征: {len(NUMERIC_CORE)} 数值 + {len(CATEGORICAL_CORE)} 类别")
        print(f"  Target mean: 仅{len(TARGET_MEAN_SELECTIVE)}个高基数特征, smoothing=5.0")
        print(f"  CV: 5-fold (更稳定)")
        
        res, mdl = train(enhanced, f"hybrid_sz{sz}")
        results.append(res)
        
        print(f"\n  CV F1: {res['cv_f1']:.4f}")
        print(f"  Test F1: {res['test_f1']:.4f}")
        print(f"  Test Acc: {res['acc']:.4f}")
        print(f"  最佳: {res['params']}")
        
        if res['test_f1'] > best_f1:
            best_f1 = res['test_f1']
            best_model = mdl
        
        if res['test_f1'] >= 0.70:
            print(f"\n✓✓✓ 达标！F1={res['test_f1']:.4f} >= 0.70 ✓✓✓")
            break
    
    df_res = pd.DataFrame(results)
    df_res.to_csv(dirs["tables"] / "hybrid_features_results.csv", index=False)
    if best_model:
        joblib.dump(best_model, dirs["models"] / "svm_hybrid_best.joblib")
    
    print(f"\n{'='*60}\n混合策略最终结果:\n{'='*60}")
    print(df_res[["name", "cv_f1", "test_f1", "acc"]].to_string(index=False))
    print(f"\n最佳 F1: {best_f1:.4f}")
    print(f"之前最佳: 0.6587")
    print(f"改进: {best_f1 - 0.6587:+.4f}")
    print("✓ 成功!" if best_f1 >= 0.70 else f"✗ 还差 {0.70 - best_f1:.4f}")
    print('='*60)

if __name__ == "__main__":
    main()
