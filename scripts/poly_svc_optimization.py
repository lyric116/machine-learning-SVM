#!/usr/bin/env python3
"""
尝试多项式核SVM
可能适合这个问题的非线性结构
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
print("多项式核SVM优化")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 精选特征
base = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

for col in ['start_station_id', 'end_station_id']:
    member_cnt = df[y==1][col].value_counts()
    total_cnt = df[col].value_counts()
    ratio = (member_cnt / total_cnt).fillna(0.5)
    df[f'{col}_mr'] = df[col].map(ratio).fillna(0.5)

df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
route_freq = df['route'].value_counts()
df['route_rare'] = df['route'].map(route_freq).apply(lambda x: 1/(x+1))

df['log_dur'] = np.log1p(df['duration_min'])

features = base + ['start_station_id_mr', 'end_station_id_mr', 'route_rare', 'log_dur']

print(f"特征数: {len(features)}")

X = df[features].replace([np.inf, -np.inf], np.nan).fillna(df[features].median())
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集: {len(X_train)}")

# 多项式核参数
param_grid = {
    'C': [0.5, 1, 5, 10],
    'degree': [2, 3],  # 2次和3次多项式
    'coef0': [0, 0.1, 1],
    'class_weight': ['balanced']
}

print(f"参数组合数: {len(param_grid['C']) * len(param_grid['degree']) * len(param_grid['coef0'])}")

svc = SVC(kernel='poly', random_state=42, cache_size=1000, max_iter=50000)

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

y_pred = grid.predict(X_test_scaled)
f1_test = f1_score(y_test, y_pred, average='weighted')
acc_test = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("结果")
print("="*70)
print(f"CV F1:   {grid.best_score_:.4f}")
print(f"Test F1: {f1_test:.4f}")
print(f"Test Acc: {acc_test:.4f}")

print("\n对比:")
print(f"  LinearSVC:     F1 = 0.6587")
print(f"  Poly SVC:      F1 = {f1_test:.4f}")
print(f"  目标:          F1 = 0.7000")

if f1_test >= 0.70:
    print(f"\n🎉 达到目标！")
elif f1_test > 0.6587:
    print(f"\n✅ 超越LinearSVC！提升 {f1_test - 0.6587:.4f}")
else:
    print(f"\n未超过LinearSVC")

pd.DataFrame([{
    'method': 'Polynomial SVC',
    'n_features': len(features),
    'best_params': str(grid.best_params_),
    'cv_f1': grid.best_score_,
    'test_f1': f1_test,
}]).to_csv('outputs/tables/poly_svc_results.csv', index=False)

print(f"\n结果已保存")
