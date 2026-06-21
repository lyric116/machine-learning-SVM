#!/usr/bin/env python3
"""Analyze why F1 is stuck at 0.65"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files, 
    profile_and_sample, clean_and_engineer)

# Load data
files = csv_files(Path("data"))
sampled, _ = profile_and_sample(files, 10000, RANDOM_STATE)
cleaned, _ = clean_and_engineer(sampled)

print("="*60)
print("DATA ANALYSIS: Why is F1 stuck at ~0.65?")
print("="*60)

# Class distribution
print("\n1. CLASS DISTRIBUTION:")
class_dist = cleaned[TARGET].value_counts()
print(class_dist)
print(f"Balance ratio: {class_dist.min() / class_dist.max():.3f}")

# Feature separability analysis
print("\n2. FEATURE SEPARABILITY (member vs casual):")
for col in ['duration_min', 'distance_km', 'speed_kmh', 'is_weekend', 'is_round_trip']:
    member_mean = cleaned[cleaned[TARGET] == 'member'][col].mean()
    casual_mean = cleaned[cleaned[TARGET] == 'casual'][col].mean()
    overall_std = cleaned[col].std()
    effect_size = abs(member_mean - casual_mean) / overall_std
    print(f"  {col:20s}  member={member_mean:7.2f}  casual={casual_mean:7.2f}  effect_size={effect_size:.3f}")

# Rideable type distribution
print("\n3. RIDEABLE TYPE BY CLASS:")
ct = pd.crosstab(cleaned['rideable_type'], cleaned[TARGET], normalize='columns') * 100
print(ct.round(1))

# Time patterns
print("\n4. TIME PATTERNS:")
print("  Weekend usage:")
weekend_ct = pd.crosstab(cleaned['is_weekend'], cleaned[TARGET], normalize='columns') * 100
print(weekend_ct.round(1))

print("\n  Commute peak:")
commute_ct = pd.crosstab(cleaned['is_commute_peak'], cleaned[TARGET], normalize='columns') * 100
print(commute_ct.round(1))

# Station missing patterns
print("\n5. STATION AVAILABILITY:")
station_ct = pd.crosstab(cleaned['start_station_missing'], cleaned[TARGET], normalize='columns') * 100
print(station_ct.round(1))

# Overlap analysis
print("\n6. FEATURE OVERLAP (how much do classes overlap?):")
# Duration bins
dur_overlap = pd.crosstab(cleaned['duration_bin'], cleaned[TARGET], normalize='columns')
overlap_score = (dur_overlap.min(axis=1) / dur_overlap.max(axis=1)).mean()
print(f"  Duration bins overlap: {overlap_score:.3f} (0=no overlap, 1=complete overlap)")

# Distance bins  
dist_overlap = pd.crosstab(cleaned['distance_bin'], cleaned[TARGET], normalize='columns')
overlap_score = (dist_overlap.min(axis=1) / dist_overlap.max(axis=1)).mean()
print(f"  Distance bins overlap: {overlap_score:.3f}")

print("\n" + "="*60)
print("CONCLUSION:")
print("If effect sizes are small (<0.3) and overlap is high (>0.5),")
print("the classes are inherently hard to separate → F1 ceiling exists.")
print("="*60)
