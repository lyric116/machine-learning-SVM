#!/usr/bin/env python3
"""系统化的特征选择和分析"""
import sys, os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from run_svm_experiment import (RANDOM_STATE, TARGET, CLASS_ORDER, csv_files,
    profile_and_sample, TargetMeanEncoder)
from svm_final_fixed import add_features, FEATURES

def analyze_features():
    print("="*60)
    print("特征选择和相关性分析")
    print("="*60)
    
    # Load data
    files = csv_files(Path("data"))
    sampled, _ = profile_and_sample(files, 10000, RANDOM_STATE)
    
    from run_svm_experiment import clean_and_engineer
    cleaned, _ = clean_and_engineer(sampled)
    enhanced = add_features(cleaned)
    
    X = enhanced[FEATURES].copy()
    y = enhanced[TARGET].copy()
    
    # Encode target
    y_encoded = (y == 'casual').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=RANDOM_STATE)
    
    # Apply target mean encoding
    tm_features = ["start_station_id", "end_station_id", "route_station_id",
                   "start_grid_ultra", "end_grid_ultra", "hour_weekend",
                   "hour_weekday", "rideable_weekend", "rideable_timeperiod"]
    
    tm_encoder = TargetMeanEncoder(columns=tm_features, smoothing=7.0, 
                                    n_splits=5, random_state=RANDOM_STATE)
    X_train_tm = tm_encoder.fit_transform(X_train, y_train)
    
    # Get numeric features only
    numeric_features = []
    for col in X_train_tm.columns:
        if X_train_tm[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_features.append(col)
    
    X_train_num = X_train_tm[numeric_features].fillna(0)
    
    print(f"\n数值特征数量: {len(numeric_features)}")
    print(f"样本数量: {len(X_train_num)}")
    
    # 1. Mutual Information分析
    print("\n" + "="*60)
    print("1. 互信息分析 (Mutual Information)")
    print("="*60)
    
    mi_scores = mutual_info_classif(X_train_num, y_train, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({
        'feature': numeric_features,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    print("\nTop 20 最重要特征:")
    print(mi_df.head(20).to_string(index=False))
    
    print("\nBottom 10 最不重要特征:")
    print(mi_df.tail(10).to_string(index=False))
    
    # 2. 特征相关性分析
    print("\n" + "="*60)
    print("2. 特征相关性分析")
    print("="*60)
    
    corr_matrix = X_train_num.corr().abs()
    
    # 找出高度相关的特征对 (>0.9)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.9:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if high_corr_pairs:
        print(f"\n发现 {len(high_corr_pairs)} 对高度相关特征 (>0.9):")
        for pair in high_corr_pairs[:10]:
            print(f"  {pair['feature1']:30s} <-> {pair['feature2']:30s}  r={pair['correlation']:.3f}")
    else:
        print("\n未发现高度相关特征对 (>0.9)")
    
    # 3. 方差分析
    print("\n" + "="*60)
    print("3. 特征方差分析")
    print("="*60)
    
    variances = X_train_num.var()
    low_var_features = variances[variances < 0.01].sort_values()
    
    print(f"\n低方差特征 (<0.01): {len(low_var_features)}个")
    if len(low_var_features) > 0:
        print(low_var_features.head(10))
    
    # 4. 推荐的特征集
    print("\n" + "="*60)
    print("4. 推荐的特征集")
    print("="*60)
    
    # Top K features by MI
    top_15 = mi_df.head(15)['feature'].tolist()
    top_20 = mi_df.head(20)['feature'].tolist()
    top_25 = mi_df.head(25)['feature'].tolist()
    
    print(f"\nTop 15 特征: {len(top_15)}个")
    print(", ".join(top_15))
    
    print(f"\nTop 20 特征: {len(top_20)}个")
    print(", ".join(top_20))
    
    print(f"\nTop 25 特征: {len(top_25)}个")
    print(", ".join(top_25))
    
    # 5. PCA分析
    print("\n" + "="*60)
    print("5. PCA降维分析")
    print("="*60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train_num)
    
    pca = PCA(n_components=20, random_state=RANDOM_STATE)
    pca.fit(X_scaled)
    
    cumsum_var = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"\nPCA主成分方差解释:")
    for i, (var, cumvar) in enumerate(zip(pca.explained_variance_ratio_[:10], cumsum_var[:10])):
        print(f"  PC{i+1:2d}: {var:6.2%}  (累计: {cumvar:6.2%})")
    
    # 找到解释95%方差需要的成分数
    n_95 = np.argmax(cumsum_var >= 0.95) + 1
    print(f"\n解释95%方差需要: {n_95}个主成分")
    
    # 6. 保存推荐
    print("\n" + "="*60)
    print("6. 特征选择建议")
    print("="*60)
    
    recommendations = {
        "top_15_features": top_15,
        "top_20_features": top_20,
        "top_25_features": top_25,
        "low_variance_features": low_var_features.index.tolist(),
        "high_corr_pairs": high_corr_pairs[:5],
        "pca_components_95": n_95,
    }
    
    import json
    with open("outputs/feature_selection_analysis.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    
    print("\n策略建议:")
    print("  1. 尝试Top 15特征 - 最精简，可能减少噪声")
    print("  2. 尝试Top 20特征 - 平衡性能和复杂度")  
    print("  3. 尝试Top 25特征 - 保留更多信息")
    print(f"  4. 尝试PCA降维到{n_95}维 - 去除相关性")
    print("\n结果已保存到 outputs/feature_selection_analysis.json")
    print("="*60)

if __name__ == "__main__":
    analyze_features()
