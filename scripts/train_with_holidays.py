#!/usr/bin/env python3
"""
使用外部数据（假日信息）重新训练SVM
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("外部数据实验：添加假日特征")
print("="*70)

# 加载带假日特征的数据
df = pd.read_csv('outputs/processed/cleaned_sample_with_holidays.csv')
y = (df['member_casual'] == 'member').astype(int)

print(f"样本数: {len(df)}")
print(f"假日样本: {df['is_federal_holiday'].sum()} ({df['is_federal_holiday'].sum()/len(df)*100:.2f}%)")

# 构建完整特征集（包括假日特征）
base = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak', 'is_weekend']

# 假日特征
holiday_feat = ['is_federal_holiday']

# 频率特征
for col in ['start_station_id', 'end_station_id']:
    member_cnt = df[y==1][col].value_counts()
    total_cnt = df[col].value_counts()
    ratio = (member_cnt / total_cnt).fillna(0.5)
    df[f'{col}_mr'] = df[col].map(ratio).fillna(0.5)

df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
route_freq = df['route'].value_counts()
df['route_rare'] = df['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq_feat = ['start_station_id_mr', 'end_station_id_mr', 'route_rare']

# 非线性变换
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['speed_x_dist'] = df['speed_kmh'] * df['distance_km']

nonlin_feat = ['log_dur', 'log_dist', 'speed_x_dist']

# 领域知识
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)

domain_feat = ['fast_commute', 'slow_leisure']

# 假日×周末交互
df['holiday_weekend'] = (df['is_federal_holiday'].astype(int) * df['is_weekend'].astype(int))

# 空间特征
if 'start_lat' in df.columns:
    center_lat, center_lng = 38.9072, -77.0369
    df['dist_center'] = np.sqrt(
        (df['start_lat'] - center_lat)**2 + (df['start_lng'] - center_lng)**2
    )
    spatial_feat = ['dist_center']
else:
    spatial_feat = []

# 所有特征
all_feat = base + holiday_feat + freq_feat + nonlin_feat + domain_feat + spatial_feat + ['holiday_weekend']

print(f"特征数: {len(all_feat)} (包含{len(holiday_feat)+1}个假日特征)")

X = df[all_feat].replace([np.inf, -np.inf], np.nan).fillna(df[all_feat].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# 标准化
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 网格搜索
print("\n训练中...")

param_grid = {
    'C': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': ['balanced']
}

clf = LinearSVC(max_iter=100000, random_state=42, dual='auto')

grid = GridSearchCV(
    clf,
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

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
print(f"  无假日特征:     F1 = 0.6587")
print(f"  有假日特征:     F1 = {f1_test:.4f}")
print(f"  提升:           {f1_test - 0.6587:+.4f}")
print(f"  目标:           F1 = 0.7000")

if f1_test >= 0.70:
    print(f"\n🎉🎉🎉 达到目标！F1 = {f1_test:.4f} >= 0.70")
elif f1_test > 0.6587:
    print(f"\n✅ 新记录！超越之前最佳")
    print(f"   距目标还差 {0.70 - f1_test:.4f}")
else:
    print(f"\n未超过之前最佳")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['casual', 'member']))

# 保存
import joblib
if f1_test > 0.6587:
    joblib.dump(grid.best_estimator_, 'outputs/models/svm_with_holidays.joblib')
    joblib.dump(scaler, 'outputs/models/svm_with_holidays_scaler.joblib')
    print(f"\n模型已保存")

pd.DataFrame([{
    'method': 'LinearSVC with Holiday Features',
    'n_features': len(all_feat),
    'has_holiday': True,
    'best_params': str(grid.best_params_),
    'cv_f1': grid.best_score_,
    'test_f1': f1_test,
    'test_acc': acc_test
}]).to_csv('outputs/tables/with_holiday_features_results.csv', index=False)

print(f"结果已保存: outputs/tables/with_holiday_features_results.csv")
