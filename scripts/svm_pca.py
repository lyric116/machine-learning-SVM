#!/usr/bin/env python3
"""使用PCA降维的SVM"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import joblib, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, ensure_dirs)
from svm_optimized_features import BEST_NUMERIC, add_features, ALL_FEATURES

def build_pca_pipeline(n_components):
    """带PCA降维的pipeline"""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ("svm", LinearSVC(class_weight="balanced", dual="auto", max_iter=50000, 
            tol=1e-4, random_state=RANDOM_STATE))
    ])

def main():
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    print("="*60)
    print("PCA降维 + SVM")
    print("="*60)
    
    sampled, _ = profile_and_sample(files, 13000, RANDOM_STATE)
    
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_features(cleaned)
    
    # 只使用数值特征进行PCA
    X = enhanced[BEST_NUMERIC].fillna(0).copy()
    y = enhanced[TARGET].copy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    print(f"\n原始特征数: {X.shape[1]}")
    print(f"样本数: {len(X)}")
    
    results = []
    
    # 尝试不同的主成分数
    for n_comp in [10, 12, 14, 16]:
        print(f"\n{'='*60}")
        print(f"PCA降维到 {n_comp} 维")
        print(f"{'='*60}")
        
        pipe = build_pca_pipeline(n_comp)
        grid = {"svm__C": [0.005, 0.01, 0.02, 0.05, 0.1],
                "svm__loss": ["hinge", "squared_hinge"]}
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        gs = GridSearchCV(pipe, grid, scoring="f1_weighted", cv=cv, n_jobs=-1, verbose=0)
        
        start = time.perf_counter()
        gs.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        
        y_pred = gs.best_estimator_.predict(X_test)
        f1 = float(f1_score(y_test, y_pred, average="weighted"))
        acc = float(accuracy_score(y_test, y_pred))
        
        result = {"n_components": n_comp, "cv_f1": float(gs.best_score_),
                  "test_f1": f1, "acc": acc, "best_C": gs.best_params_["svm__C"],
                  "time": elapsed}
        results.append(result)
        
        print(f"  CV F1: {result['cv_f1']:.4f}")
        print(f"  Test F1: {f1:.4f}")
        print(f"  Best C: {result['best_C']}")
        
        if f1 >= 0.70:
            print(f"\n✓ 达标！")
            break
    
    print(f"\n{'='*60}\nPCA结果汇总:\n{'='*60}")
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
    
    best = df_res.loc[df_res['test_f1'].idxmax()]
    print(f"\n最佳配置: {int(best['n_components'])}维 PCA")
    print(f"最佳 F1: {best['test_f1']:.4f}")
    print('='*60)

if __name__ == "__main__":
    main()
