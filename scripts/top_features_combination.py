#!/usr/bin/env python3
"""
Top特征组合测试
测试Top-N特征组合的效果
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

print("Top特征渐进式组合测试")
print("="*70)

# 加载数据
df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# Top特征列表（基于单特征测试）
top_features = [
    'speed_kmh',
    'weekday',
    'duration_min',
    'is_weekend',
    'end_lat',
    'start_lat',
    'is_commute_peak',
    'end_lng',
    'start_lng',
    'hour'
]

print(f"测试Top-{len(top_features)}特征的渐进组合")
print()

results = []

for n in range(1, len(top_features) + 1):
    # 选择前n个特征
    selected_features = top_features[:n]

    X = df[selected_features].copy()
    X = X.fillna(X.median())

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM（使用最佳配置）
    clf = LinearSVC(C=0.003, loss='hinge', class_weight='balanced',
                   max_iter=50000, random_state=42, dual='auto')
    clf.fit(X_train_scaled, y_train)

    # 评估
    y_pred = clf.predict(X_test_scaled)
    f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    results.append({
        'n_features': n,
        'features': ', '.join(selected_features),
        'f1': f1,
        'accuracy': acc
    })

    print(f"Top-{n:2d}: F1={f1:.4f}  (+{selected_features[-1]})")

# 对比完整模型
print()
print("="*70)
print("对比分析:")
print("="*70)
print(f"Top-1特征:    F1 = {results[0]['f1']:.4f}  (speed_kmh)")
print(f"Top-3特征:    F1 = {results[2]['f1']:.4f}  (speed + weekday + duration)")
print(f"Top-5特征:    F1 = {results[4]['f1']:.4f}")
print(f"Top-10特征:   F1 = {results[-1]['f1']:.4f}")
print(f"完整模型(37): F1 = 0.6587")
print()

# 计算边际贡献
print("边际贡献分析:")
print("-"*70)
for i in range(len(results)):
    if i == 0:
        contribution = results[i]['f1']
        print(f"第{i+1}个特征: +{contribution:.4f}  (基线)")
    else:
        contribution = results[i]['f1'] - results[i-1]['f1']
        print(f"第{i+1}个特征: +{contribution:.4f}  ({top_features[i]})")

# 保存结果
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/tables/top_features_combination.csv', index=False)
print()
print("结果已保存: outputs/tables/top_features_combination.csv")

# 关键洞察
print()
print("="*70)
print("🔍 关键洞察:")
print("="*70)

top3_f1 = results[2]['f1']
top10_f1 = results[-1]['f1']
full_f1 = 0.6587

improvement_3_to_10 = top10_f1 - top3_f1
improvement_10_to_37 = full_f1 - top10_f1

print(f"1. Top-3特征即可达到 F1={top3_f1:.4f}")
print(f"2. Top-10特征达到 F1={top10_f1:.4f}，距完整模型仅差 {full_f1-top10_f1:.4f}")
print(f"3. 从Top-3到Top-10的提升: {improvement_3_to_10:.4f}")
print(f"4. 从Top-10到完整37特征的提升: {improvement_10_to_37:.4f}")
print()
print(f"💡 结论: {'前10个特征贡献了大部分性能' if improvement_10_to_37 < 0.02 else '需要更多特征协同'}")
