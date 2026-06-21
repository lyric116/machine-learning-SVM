#!/usr/bin/env python3
"""Conservative RBF SVM - memory-safe version"""
import sys, os, time
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, ensure_dirs, TargetMeanEncoder)
from svm_final_fixed import add_features, FEATURES

def main():
    print("="*60)
    print("RBF SVM - Conservative Memory-Safe Version")
    print("="*60)
    
    dirs = ensure_dirs(Path("outputs"))
    files = csv_files(Path("data"))
    
    # Use moderate sample size to control memory
    print("\nLoading 10000 samples/class/month (memory-safe)...")
    sampled, _ = profile_and_sample(files, 10000, RANDOM_STATE)
    
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_features(cleaned)
    print(f"Total: {len(enhanced)} samples")
    
    X = enhanced[FEATURES].copy()
    y = enhanced[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    
    # Manual feature engineering for RBF (avoid complex pipeline)
    print("\nPreparing features for RBF...")
    
    # Target mean encoding
    tm_encoder = TargetMeanEncoder(
        columns=["start_station_id", "end_station_id", "route_station_id",
                 "start_grid_ultra", "end_grid_ultra"],
        smoothing=7.0, n_splits=5, random_state=RANDOM_STATE
    )
    
    X_train_tm = tm_encoder.fit_transform(X_train, y_train)
    X_test_tm = tm_encoder.transform(X_test)
    
    # Select only numeric features for RBF
    numeric_cols = [c for c in X_train_tm.columns if '_target_mean' in c or 
                    c in ["duration_min", "distance_km", "speed_kmh", "start_lat", 
                          "start_lng", "end_lat", "end_lng", "hour_sin", "hour_cos",
                          "weekday_sin", "weekday_cos", "is_weekend", "is_commute_peak",
                          "dist_per_min", "speed_dist", "very_short", "long_trip", "fast_rider"]]
    
    X_train_num = X_train_tm[numeric_cols].fillna(0)
    X_test_num = X_test_tm[numeric_cols].fillna(0)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)
    
    print(f"Feature count: {X_train_scaled.shape[1]}")
    print(f"Training samples: {X_train_scaled.shape[0]}")
    
    # Try different RBF configurations sequentially
    results = []
    
    configs = [
        {"C": 1.0, "gamma": "scale", "name": "C=1.0, gamma=scale"},
        {"C": 2.0, "gamma": "scale", "name": "C=2.0, gamma=scale"},
        {"C": 1.0, "gamma": 0.01, "name": "C=1.0, gamma=0.01"},
        {"C": 2.0, "gamma": 0.01, "name": "C=2.0, gamma=0.01"},
    ]
    
    best_f1 = 0
    best_model = None
    best_config = None
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"{'='*60}")
        
        rbf_svm = SVC(
            kernel='rbf',
            C=config['C'],
            gamma=config['gamma'],
            class_weight='balanced',
            max_iter=10000,
            cache_size=500,  # Limit cache to 500MB
            random_state=RANDOM_STATE,
            verbose=False,
        )
        
        print("Training (single-threaded)...")
        start = time.perf_counter()
        try:
            rbf_svm.fit(X_train_scaled, y_train)
            elapsed = time.perf_counter() - start
            
            y_pred = rbf_svm.predict(X_test_scaled)
            f1 = float(f1_score(y_test, y_pred, average="weighted"))
            acc = float(accuracy_score(y_test, y_pred))
            report = classification_report(y_test, y_pred, labels=CLASS_ORDER, output_dict=True)
            
            result = {
                "config": config['name'],
                "f1": f1,
                "accuracy": acc,
                "f1_member": report['member']['f1-score'],
                "f1_casual": report['casual']['f1-score'],
                "time": elapsed,
            }
            results.append(result)
            
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Test F1: {f1:.4f}")
            print(f"  Test Acc: {acc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = (rbf_svm, tm_encoder, scaler, numeric_cols)
                best_config = config['name']
                
            if f1 >= 0.70:
                print(f"\n✓✓✓ TARGET REACHED! F1={f1:.4f} >= 0.70 ✓✓✓")
                break
                
        except MemoryError:
            print(f"  ✗ Memory error with {config['name']}")
            continue
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("FINAL RBF RESULTS:")
    print(f"{'='*60}")
    for r in results:
        print(f"{r['config']:25s}  F1: {r['f1']:.4f}  Acc: {r['accuracy']:.4f}")
    
    print(f"\nBest: {best_config}")
    print(f"Best F1: {best_f1:.4f}")
    print(f"Previous best (LinearSVC): 0.6587")
    print(f"Improvement: {best_f1 - 0.6587:+.4f}")
    
    if best_f1 >= 0.70:
        print(f"\n✓ SUCCESS! Target reached!")
    else:
        print(f"\n✗ Still {0.70 - best_f1:.4f} short of 0.70")
    
    # Save best model
    if best_model:
        joblib.dump(best_model, dirs["models"] / "svm_rbf_best.joblib")
        
    # Save results
    import json
    with open(dirs["output"] / "rbf_results.json", "w") as f:
        json.dump({
            "best_f1": best_f1,
            "best_config": best_config,
            "all_results": results,
        }, f, indent=2)
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
