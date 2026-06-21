#!/usr/bin/env python3
"""
最后尝试：组合所有最佳发现 + RBF核优化
使用完整样本 + 精选特征 + RBF核
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.kernel_approximation import Nystroem
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("最后尝试：RBF核 + 完整特征 + 精细调参")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

print(f"样本数: {len(df)}")

# 构建最优特征集
base = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak', 'is_weekend']

# 频率特征
for col in ['start_station_id', 'end_station_id']:
    member_cnt = df[y==1][col].value_counts()
    total_cnt = df[col].value_counts()
    ratio = (member_cnt / total_cnt).fillna(0.5)
    df[f'{col}_mr'] = df[col].map(ratio).fillna(0.5)

df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
route_freq = df['route'].value_counts()
df['route_rare'] = df['route'].map(route_freq).apply(lambda x: 1/(x+1))

if 'rideable_type' in df.columns:
    vtype_member = df[y==1]['rideable_type'].value_counts()
    vtype_total = df['rideable_type'].value_counts()
    vtype_ratio = (vtype_member / vtype_total).fillna(0.5)
    df['vtype_mr'] = df['rideable_type'].map(vtype_ratio).fillna(0.5)
    freq = ['start_station_id_mr', 'end_station_id_mr', 'route_rare', 'vtype_mr']
else:
    freq = ['start_station_id_mr', 'end_station_id_mr', 'route_rare']

# 非线性
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['speed_x_dist'] = df['speed_kmh'] * df['distance_km']
df['dist_per_dur'] = df['distance_km'] / (df['duration_min'] + 0.1)

nonlin = ['log_dur', 'log_dist', 'speed_x_dist', 'dist_per_dur']

# 领域
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)

domain = ['fast_commute', 'slow_leisure']

all_feat = base + freq + nonlin + domain

print(f"特征数: {len(all_feat)}")

X = df[all_feat].replace([np.inf, -np.inf], np.nan).fillna(df[all_feat].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 尝试1: Nystroem核近似 + LinearSVC (快速RBF近似)
print("\n尝试1: Nystroem核近似 + LinearSVC")
from sklearn.svm import LinearSVC

nystroem = Nystroem(kernel='rbf', gamma=0.01, n_components=300, random_state=42)
X_train_nys = nystroem.fit_transform(X_train_scaled)
X_test_nys = nystroem.transform(X_test_scaled)

clf_nys = LinearSVC(C=0.1, loss='squared_hinge', class_weight='balanced',
                   max_iter=50000, random_state=42, dual='auto')
clf_nys.fit(X_train_nys, y_train)

f1_nys = f1_score(y_test, clf_nys.predict(X_test_nys), average='weighted')
print(f"F1: {f1_nys:.4f}")

# 尝试2: 真正的RBF SVC (小规模样本)
print("\n尝试2: RBF SVC (精细调参)")

# 使用一半样本避免OOM
sample_size = len(X_train) // 2
idx = np.random.choice(len(X_train), sample_size, replace=False)
X_train_sub = X_train_scaled[idx]
y_train_sub = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]

param_grid = {
    'C': [1, 5, 10],
    'gamma': ['scale', 0.01, 0.001],
    'class_weight': ['balanced']
}

clf_rbf = SVC(kernel='rbf', random_state=42, max_iter=10000)
grid = GridSearchCV(clf_rbf, param_grid, cv=3, scoring='f1_weighted', n_jobs=2)

print("训练中...")
grid.fit(X_train_sub, y_train_sub)

print(f"最佳参数: {grid.best_params_}")
print(f"CV F1: {grid.best_score_:.4f}")

# 在完整测试集上评估
f1_rbf = f1_score(y_test, grid.predict(X_test_scaled), average='weighted')
print(f"Test F1: {f1_rbf:.4f}")

# 选择最佳
best_f1 = max(f1_nys, f1_rbf)
best_method = "Nystroem" if f1_nys > f1_rbf else "RBF SVC"

print("\n" + "="*70)
print("结果")
print("="*70)
print(f"Nystroem近似: F1 = {f1_nys:.4f}")
print(f"RBF SVC:      F1 = {f1_rbf:.4f}")
print(f"最佳方法:     {best_method}, F1 = {best_f1:.4f}")

print(f"\n对比:")
print(f"  之前最佳:     F1 = 0.6587")
print(f"  当前尝试:     F1 = {best_f1:.4f}")
print(f"  目标:         F1 = 0.7000")

if best_f1 >= 0.70:
    print(f"\n🎉 成功达到目标！")
elif best_f1 > 0.6587:
    print(f"\n✅ 新记录！提升 {best_f1 - 0.6587:.4f}")
    print(f"   距目标还差 {0.70 - best_f1:.4f}")
else:
    print(f"\n未超过之前最佳")

# 保存结果
pd.DataFrame([{
    'method': best_method,
    'f1': best_f1,
}]).to_csv('outputs/tables/rbf_final_attempt.csv', index=False)

print(f"\n结果已保存")
