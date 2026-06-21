#!/usr/bin/env python3
"""
创新思路8: 难样本挖掘 + 样本加权
关注那些难以分类的边界样本
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("创新思路8: 难样本挖掘 + 迭代优化")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 构建完整特征集
df_full = df.copy()

# 基础 + 频率特征
base_features = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

for col in ['start_station_id', 'end_station_id']:
    member_count = df[y==1][col].value_counts()
    total_count = df[col].value_counts()
    member_ratio = (member_count / total_count).fillna(0.5)
    df_full[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)

df_full['route'] = df['start_station_id'].astype(str) + '_TO_' + df['end_station_id'].astype(str)
route_freq = df_full['route'].value_counts()
df_full['route_rarity'] = df_full['route'].map(route_freq).apply(lambda x: 1/(x+1))

# 领域特征
df_full['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df_full['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)
df_full['efficiency'] = df['distance_km'] / (df['duration_min'] + 1)

all_features = base_features + [
    'start_station_id_member_ratio', 'end_station_id_member_ratio', 'route_rarity',
    'fast_commute', 'slow_leisure', 'efficiency'
]

X = df_full[all_features].fillna(df_full[all_features].median())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 策略1: 难样本加权 ====================
print("\n策略1: 识别并加权难样本")

# 第一轮训练
clf1 = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
                max_iter=50000, random_state=42, dual='auto')
clf1.fit(X_train_scaled, y_train)

# 获取决策函数值（距离超平面的距离）
decision_values = np.abs(clf1.decision_function(X_train_scaled))

# 难样本：决策值小（接近决策边界）
threshold = np.percentile(decision_values, 30)  # 最难的30%
hard_samples = decision_values < threshold

# 构建样本权重
sample_weights = np.ones(len(y_train))
sample_weights[hard_samples] = 2.0  # 难样本权重加倍

# 第二轮训练（加权）
clf2 = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
                max_iter=50000, random_state=42, dual='auto')
clf2.fit(X_train_scaled, y_train, sample_weight=sample_weights)

f1_weighted = f1_score(y_test, clf2.predict(X_test_scaled), average='weighted')
print(f"F1: {f1_weighted:.4f}")

# ==================== 策略2: 迭代难样本挖掘 ====================
print("\n策略2: 多轮迭代优化")

sample_weights = np.ones(len(y_train))

for iteration in range(3):
    # 训练
    clf_iter = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
                        max_iter=50000, random_state=42+iteration, dual='auto')
    clf_iter.fit(X_train_scaled, y_train, sample_weight=sample_weights)

    # 预测
    y_pred_train = clf_iter.predict(X_train_scaled)

    # 找出错误样本
    errors = (y_pred_train != y_train)

    # 增加错误样本权重
    sample_weights[errors] *= 1.5

    print(f"  轮{iteration+1}: 错误样本 {errors.sum()} ({errors.sum()/len(y_train)*100:.1f}%)")

f1_iter = f1_score(y_test, clf_iter.predict(X_test_scaled), average='weighted')
print(f"F1: {f1_iter:.4f}")

# ==================== 策略3: 混合预测 ====================
print("\n策略3: 多模型混合")

# 训练多个随机初始化的模型
predictions = []
for seed in [42, 43, 44, 45, 46]:
    clf_mix = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
                       max_iter=50000, random_state=seed, dual='auto')
    clf_mix.fit(X_train_scaled, y_train)
    pred = clf_mix.predict(X_test_scaled)
    predictions.append(pred)

# 投票
predictions = np.array(predictions)
y_pred_vote = (predictions.sum(axis=0) >= 3).astype(int)  # 多数投票

f1_vote = f1_score(y_test, y_pred_vote, average='weighted')
print(f"F1: {f1_vote:.4f}")

# ==================== 总结 ====================
print("\n" + "="*70)
print("结果总结")
print("="*70)

baseline = 0.6271
results = [
    ('Baseline (8特征+频率)', baseline),
    ('策略1: 难样本加权', f1_weighted),
    ('策略2: 迭代优化', f1_iter),
    ('策略3: 多模型投票', f1_vote)
]

for name, f1 in results:
    print(f"{name:25s} F1={f1:.4f} ({f1-baseline:+.4f})")

best_name, best_f1 = max(results[1:], key=lambda x: x[1])
print(f"\n最佳: {best_name}, F1={best_f1:.4f}")

if best_f1 >= 0.70:
    print(f"\n🎉 达到目标 F1>=0.70!")
elif best_f1 >= 0.65:
    print(f"\n✅ 接近目标，还差 {0.70 - best_f1:.4f}")

pd.DataFrame(results, columns=['Method', 'F1']).to_csv(
    'outputs/tables/hard_sample_mining_results.csv', index=False)

print(f"\n结果已保存: outputs/tables/hard_sample_mining_results.csv")
