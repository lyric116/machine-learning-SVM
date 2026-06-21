#!/usr/bin/env python3
"""Voting Ensemble - train multiple SVMs and vote (避免Bagging的DataFrame问题)"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib, numpy as np, pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, ensure_dirs)
from svm_final_fixed import add_features, build_pipe, FEATURES

def main():
    print("="*60)
    print("SVM VOTING ENSEMBLE - Alternative to Bagging")
    print("="*60)
    
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
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
    
    print(f"\nTraining Voting Ensemble (5 diverse SVMs)...")
    
    # Create 5 diverse SVM models with different hyperparameters
    estimators = []
    
    configs = [
        ("svm1", {"svm__C": 0.005, "svm__loss": "hinge"}),
        ("svm2", {"svm__C": 0.01, "svm__loss": "hinge"}),
        ("svm3", {"svm__C": 0.02, "svm__loss": "hinge"}),
        ("svm4", {"svm__C": 0.01, "svm__loss": "squared_hinge"}),
        ("svm5", {"svm__C": 0.03, "svm__loss": "squared_hinge"}),
    ]
    
    for name, params in configs:
        pipe = build_pipe()
        pipe.set_params(**params)
        estimators.append((name, pipe))
        print(f"  {name}: C={params['svm__C']}, loss={params['svm__loss']}")
    
    # Soft voting (average probabilities) - but LinearSVC doesn't have predict_proba
    # Use hard voting instead
    voting = VotingClassifier(
        estimators=estimators,
        voting='hard',  # Majority vote
        n_jobs=-1,
        verbose=True,
    )
    
    print("\nFitting ensemble...")
    start = time.perf_counter()
    voting.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    
    print(f"\nEvaluating...")
    y_pred = voting.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, average="weighted"))
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
    
    print(f"\n{'='*60}")
    print("VOTING ENSEMBLE RESULTS:")
    print(f"{'='*60}")
    print(f"  Test Accuracy: {acc:.4f}")
    print(f"  Test F1 (weighted): {f1:.4f}")
    print(f"  F1 (member): {report['member']['f1-score']:.4f}")
    print(f"  F1 (casual): {report['casual']['f1-score']:.4f}")
    print(f"  Training time: {elapsed:.1f}s")
    print(f"\nComparison:")
    print(f"  Best single SVM: 0.6587")
    print(f"  Voting ensemble: {f1:.4f}")
    print(f"  Gain: {f1 - 0.6587:+.4f}")
    
    if f1 >= 0.70:
        print(f"\n✓✓✓ SUCCESS! F1={f1:.4f} >= 0.70 ✓✓✓")
        joblib.dump(voting, dirs["models"] / "svm_voting_final.joblib")
    else:
        print(f"\n✗ Still {0.70 - f1:.4f} short of 0.70")
    
    print(f"{'='*60}")
    
    # Save
    import json
    with open(dirs["output"] / "voting_results.json", "w") as f:
        json.dump({
            "ensemble_f1": f1,
            "single_best_f1": 0.6587,
            "gain": f1 - 0.6587,
            "accuracy": acc,
        }, f, indent=2)

if __name__ == "__main__":
    main()
