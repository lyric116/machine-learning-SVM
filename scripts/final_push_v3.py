#!/usr/bin/env python3
"""
最终冲刺：结合Kaggle最佳实践的全新方案
灵感来源：
1. 使用更大的样本量（之前只用了13k/class/month）
2. 更激进的特征组合
3. 数据增强技术
4. 多尺度特征
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("终极方案：使用更大样本量 + 最优特征组合")
print("="*70)

# ==================== 策略1: 使用更大的样本量 ====================
print("\n策略1: 加载更大样本（尝试20k/class/month）")

df_raw = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
print(f"当前清洗后样本: {len(df_raw)}")

# 如果样本太少，尝试重新采样更多
# 这里我们使用已有的全部样本
df = df_raw.copy()
y = (df['member_casual'] == 'member').astype(int)

print(f"使用样本数: {len(df)}")
print(f"会员: {y.sum()}, 散户: {(1-y).sum()}")

# ==================== 策略2: 构建全面的特征集 ====================
print("\n策略2: 构建全面特征集")

# 基础特征
base_feat = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 频率特征
print("  - 频率特征...")
for col in ['start_station_id', 'end_station_id']:
    member_cnt = df[y==1][col].value_counts()
    total_cnt = df[col].value_counts()
    ratio = (member_cnt / total_cnt).fillna(0.5)
    df[f'{col}_mratio'] = df[col].map(ratio).fillna(0.5)

df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
route_freq = df['route'].value_counts()
df['route_rare'] = df['route'].map(route_freq).apply(lambda x: 1/(x+1))

# 车辆类型频率
if 'rideable_type' in df.columns:
    vtype_member = df[y==1]['rideable_type'].value_counts()
    vtype_total = df['rideable_type'].value_counts()
    vtype_ratio = (vtype_member / vtype_total).fillna(0.5)
    df['vtype_mratio'] = df['rideable_type'].map(vtype_ratio).fillna(0.5)

freq_feat = ['start_station_id_mratio', 'end_station_id_mratio', 'route_rare', 'vtype_mratio']

# 非线性变换
print("  - 非线性变换...")
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['log_speed'] = np.log1p(df['speed_kmh'])
df['speed2'] = df['speed_kmh'] ** 2
df['dur2'] = df['duration_min'] ** 2
df['speed_x_dist'] = df['speed_kmh'] * df['distance_km']
df['speed_x_dur'] = df['speed_kmh'] * df['duration_min']
df['dist_per_dur'] = df['distance_km'] / (df['duration_min'] + 0.1)

nonlin_feat = ['log_dur', 'log_dist', 'log_speed', 'speed2', 'dur2',
               'speed_x_dist', 'speed_x_dur', 'dist_per_dur']

# 领域知识
print("  - 领域知识...")
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)
df['weekend_long'] = ((df['is_weekend'] == 1) & (df['duration_min'] > 20)).astype(int)

if 'hour' in df.columns:
    df['morning_fast'] = ((df['hour'].between(7, 9)) & (df['speed_kmh'] > 12)).astype(int)
    df['evening_fast'] = ((df['hour'].between(17, 19)) & (df['speed_kmh'] > 12)).astype(int)
    df['lunch_time'] = df['hour'].between(11, 14).astype(int)

domain_feat = ['fast_commute', 'slow_leisure', 'weekend_long',
               'morning_fast', 'evening_fast', 'lunch_time']

# 空间特征
print("  - 空间特征...")
if 'start_lat' in df.columns:
    # 到市中心的距离（假设DC市中心在38.9072, -77.0369）
    center_lat, center_lng = 38.9072, -77.0369
    df['dist_to_center_start'] = np.sqrt(
        (df['start_lat'] - center_lat)**2 + (df['start_lng'] - center_lng)**2
    )
    df['dist_to_center_end'] = np.sqrt(
        (df['end_lat'] - center_lat)**2 + (df['end_lng'] - center_lng)**2
    )

    # 南北东西方向
    df['heading_ns'] = df['end_lat'] - df['start_lat']  # 北为正
    df['heading_ew'] = df['end_lng'] - df['start_lng']  # 东为正

    spatial_feat = ['dist_to_center_start', 'dist_to_center_end', 'heading_ns', 'heading_ew']
else:
    spatial_feat = []

# 组合所有特征
all_feat = base_feat + freq_feat + nonlin_feat + domain_feat + spatial_feat

print(f"\n总特征数: {len(all_feat)}")

# ==================== 策略3: 数据准备 ====================
X = df[all_feat].copy()
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# 使用RobustScaler（对异常值更鲁棒）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ==================== 策略4: 更激进的超参数搜索 ====================
print("\n策略4: 激进超参数搜索")

scaler = RobustScaler()  # 使用RobustScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 扩大搜索范围
param_grid = {
    'C': [0.003, 0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': ['balanced', None],
    'max_iter': [50000]
}

print("参数候选数:", len(param_grid['C']) * len(param_grid['loss']) * len(param_grid['class_weight']))

grid = GridSearchCV(
    LinearSVC(random_state=42, dual='auto'),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
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
print("📊 最终结果")
print("="*70)
print(f"特征数: {len(all_feat)}")
print(f"CV F1:   {grid.best_score_:.4f}")
print(f"Test F1: {f1_test:.4f}")
print(f"Test Acc: {acc_test:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['casual', 'member']))

print("\n历史对比:")
print(f"  18特征+非线性:  F1 = 0.6308")
print(f"  完整37特征:     F1 = 0.6587")
print(f"  当前方案:       F1 = {f1_test:.4f}")

if f1_test >= 0.70:
    print(f"\n🎉🎉🎉 成功！F1 = {f1_test:.4f} >= 0.70")
elif f1_test >= 0.68:
    print(f"\n🎊 非常接近！仅差 {0.70 - f1_test:.4f}")
elif f1_test >= 0.65:
    print(f"\n✅ 显著改进！还差 {0.70 - f1_test:.4f}")
else:
    print(f"\n继续探索其他方向...")

# 保存最佳模型
import joblib
if f1_test >= 0.65:
    joblib.dump(grid.best_estimator_, 'outputs/models/final_push_best.joblib')
    joblib.dump(scaler, 'outputs/models/final_push_scaler.joblib')
    print(f"\n模型已保存: outputs/models/final_push_best.joblib")

# 保存结果
pd.DataFrame([{
    'method': 'Final Push - Large Sample + All Features',
    'n_features': len(all_feat),
    'n_samples': len(df),
    'cv_f1': grid.best_score_,
    'test_f1': f1_test,
    'test_acc': acc_test,
    'best_params': str(grid.best_params_)
}]).to_csv('outputs/tables/final_push_results.csv', index=False)

print(f"\n结果已保存: outputs/tables/final_push_results.csv")
