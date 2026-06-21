#!/usr/bin/env python3
"""SVM Bagging Ensemble - final attempt to reach 0.70"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, ensure_dirs)
from svm_final_fixed import add_features, build_pipe, FEATURES

def main():
    print("="*60)
    print("SVM BAGGING ENSEMBLE - Final Push to 0.70")
    print("="*60)
    
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    # Use best performing sample size (13k)
    print("\nLoading 13000 samples/class/month...")
    sampled, _ = profile_and_sample(files, 13000, RANDOM_STATE)
    
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_features(cleaned)
    print(f"Total: {len(enhanced)} samples")
    
    X = enhanced[FEATURES].copy()
    y = enhanced[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    # Build bagging ensemble
    print(f"\nTraining SVM Bagging Ensemble...")
    print(f"  Base: LinearSVC pipeline (C=0.01, max_iter=50000)")
    print(f"  Ensemble: 15 estimators, bootstrap=True")
    
    base = build_pipe()
    # Set best C from grid search
    base.set_params(svm__C=0.01, svm__loss='hinge')
    
    bagging = BaggingClassifier(
        estimator=base,
        n_estimators=15,
        max_samples=0.85,
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
    print("FINAL RESULTS:")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(f"  F1 (member): {report['member']['f1-score']:.4f}")
    print(f"  F1 (casual): {report['casual']['f1-score']:.4f}")
    print(f"  Training time: {train_time:.1f}s")
    print(f"{'='*60}")
    
    if f1 >= 0.70:
        print(f"\n✓✓✓ SUCCESS! F1={f1:.4f} >= 0.70 ✓✓✓")
        joblib.dump(bagging, dirs["models"] / "svm_bagging_final.joblib")
    else:
        print(f"\n✗ Still {0.70 - f1:.4f} short of 0.70")
        print(f"Best single model was 0.6587, ensemble is {f1:.4f}")
        print(f"Gain from ensemble: {f1 - 0.6587:.4f}")
    
    # Save results
    result = {
        "model": "SVM Bagging (15 estimators)",
        "base_f1": 0.6587,
        "ensemble_f1": f1,
        "gain": f1 - 0.6587,
        "accuracy": acc,
        "f1_member": report['member']['f1-score'],
        "f1_casual": report['casual']['f1-score'],
    }
    
    import json
    with open(dirs["output"] / "bagging_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
