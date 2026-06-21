#!/usr/bin/env python3
"""SVM Bagging Ensemble - still uses SVM but with ensemble power"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, ensure_dirs)
from optimize_svm_final import add_aggressive_features, build_pipeline, MODEL_FEATURES

def main():
    print("="*60)
    print("SVM BAGGING ENSEMBLE (still SVM-based!)")
    print("="*60)
    
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    # Use large sample
    print("\nLoading 15000 samples/class/month...")
    sampled, _ = profile_and_sample(files, 15000, RANDOM_STATE)
    
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_aggressive_features(cleaned)
    print(f"Total samples: {len(enhanced)}")
    
    X = enhanced[MODEL_FEATURES].copy()
    y = enhanced[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    print(f"\nTraining SVM Bagging Ensemble...")
    print(f"  Base: LinearSVC with C=0.01")
    print(f"  Ensemble: 10 estimators with bootstrap sampling")
    
    # Get the full pipeline
    base_pipeline = build_pipeline()
    
    # Wrap in BaggingClassifier
    bagging = BaggingClassifier(
        estimator=base_pipeline,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.9,
        bootstrap=True,
        bootstrap_features=False,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    start = time.perf_counter()
    bagging.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    
    y_pred = bagging.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(f"  F1 (member): {report['member']['f1-score']:.4f}")
    print(f"  F1 (casual): {report['casual']['f1-score']:.4f}")
    print(f"  Training time: {train_time:.1f}s")
    
    if f1 >= 0.70:
        print(f"\n✓✓✓ SUCCESS! F1={f1:.4f} >= 0.70 ✓✓✓")
        joblib.dump(bagging, dirs["models"] / "svm_bagging_ensemble.joblib")
    else:
        print(f"\n✗ Still {0.70 - f1:.4f} short of 0.70")
    print('='*60)

if __name__ == "__main__":
    main()
