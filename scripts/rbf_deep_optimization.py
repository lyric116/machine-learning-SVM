#!/usr/bin/env python3
"""
深度RBF SVM优化
系统化地探索RBF核的潜力
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("深度RBF SVM优化")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 构建最优特征集（基于之前的发现）
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

# 非线性变换
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['speed_x_dist'] = df['speed_kmh'] * df['distance_km']

# 领域知识
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)

# 精选特征（避免维度灾难）
features = base + ['start_station_id_mr', 'end_station_id_mr', 'route_rare',
                  'log_dur', 'log_dist', 'speed_x_dist', 'fast_commute']

print(f"特征数: {len(features)}")

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())

# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 标准化（RBF对尺度敏感）
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n策略1: 使用全部训练数据 + 更广泛的参数搜索")

# 扩大参数搜索范围
param_grid = {
    'C': [0.5, 1, 2, 5, 10, 20],
    'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 'scale', 'auto'],
    'class_weight': ['balanced']
}

print(f"参数组合数: {len(param_grid['C']) * len(param_grid['gamma'])}")

svc = SVC(kernel='rbf', random_state=42, cache_size=1000, max_iter=50000)

grid = GridSearchCV(
    svc,
    param_grid,
    cv=StratifiedKFold(3, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=3,
    verbose=2
)

print("开始训练...")
grid.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳CV F1: {grid.best_score_:.4f}")

# 测试集评估
y_pred = grid.predict(X_test_scaled)
f1_test = f1_score(y_test, y_pred, average='weighted')
acc_test = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("最终结果")
print("="*70)
print(f"CV F1:   {grid.best_score_:.4f}")
print(f"Test F1: {f1_test:.4f}")
print(f"Test Acc: {acc_test:.4f}")

print("\n对比:")
print(f"  LinearSVC最佳:  F1 = 0.6587")
print(f"  RBF SVC:        F1 = {f1_test:.4f}")
print(f"  目标:           F1 = 0.7000")

if f1_test >= 0.70:
    print(f"\n🎉🎉🎉 成功达到目标！F1 = {f1_test:.4f}")
elif f1_test > 0.6587:
    print(f"\n🎊 超越LinearSVC！提升 {f1_test - 0.6587:.4f}")
    print(f"   距目标还差 {0.70 - f1_test:.4f}")
elif f1_test > 0.65:
    print(f"\n✅ 有改进，距目标 {0.70 - f1_test:.4f}")
else:
    print(f"\n未超过LinearSVC")

# 保存结果
import joblib
if f1_test > 0.6587:
    joblib.dump(grid.best_estimator_, 'outputs/models/rbf_svm_best.joblib')
    joblib.dump(scaler, 'outputs/models/rbf_svm_scaler.joblib')
    print(f"\n模型已保存")

pd.DataFrame([{
    'method': 'RBF SVC Deep Optimization',
    'n_features': len(features),
    'best_params': str(grid.best_params_),
    'cv_f1': grid.best_score_,
    'test_f1': f1_test,
    'test_acc': acc_test
}]).to_csv('outputs/tables/rbf_deep_optimization.csv', index=False)

print(f"结果已保存: outputs/tables/rbf_deep_optimization.csv")
