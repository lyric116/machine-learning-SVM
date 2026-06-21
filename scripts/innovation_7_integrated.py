#!/usr/bin/env python3
"""
创新思路7: 组合最佳方案
整合所有有效的创新特征 + 精细调参
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("创新思路7: 整合所有有效创新 + 精细调参")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

print(f"样本数: {len(df)}")

# ==================== 构建所有有效特征 ====================
df_full = df.copy()

# 1. 基础特征
base_features = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 2. 频率特征
print("\n构建频率特征...")
for col in ['start_station_id', 'end_station_id']:
    member_count = df[y==1][col].value_counts()
    total_count = df[col].value_counts()
    member_ratio = (member_count / total_count).fillna(0.5)
    df_full[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)

df_full['route'] = df['start_station_id'].astype(str) + '_TO_' + df['end_station_id'].astype(str)
route_freq = df_full['route'].value_counts()
df_full['route_rarity'] = df_full['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq_features = ['start_station_id_member_ratio', 'end_station_id_member_ratio', 'route_rarity']

# 3. 领域知识交互特征
print("构建领域知识特征...")
df_full['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df_full['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)
df_full['efficiency'] = df['distance_km'] / (df['duration_min'] + 1)
df_full['weekend_long'] = ((df['is_weekend'] == 1) & (df['duration_min'] > 20)).astype(int)

if 'hour' in df.columns:
    df_full['morning_rush_fast'] = ((df['hour'].between(7, 9)) & (df['speed_kmh'] > 12)).astype(int)
    df_full['evening_rush_fast'] = ((df['hour'].between(17, 19)) & (df['speed_kmh'] > 12)).astype(int)

df_full['weekend_rare_route'] = df_full['route_rarity'] * df['is_weekend']

domain_features = ['fast_commute', 'slow_leisure', 'efficiency', 'weekend_long',
                  'morning_rush_fast', 'evening_rush_fast', 'weekend_rare_route']

# 4. 时间模式特征
print("构建时间模式特征...")
if 'hour' in df.columns:
    df_full['early_morning'] = df['hour'].between(5, 7).astype(int)
    df_full['midday'] = df['hour'].between(10, 16).astype(int)
    df_full['night'] = df['hour'].between(20, 23).astype(int)
    df_full['rush_speed_score'] = (
        ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19))) *
        (df['speed_kmh'] > 12).astype(int)
    )

time_features = ['early_morning', 'midday', 'night', 'rush_speed_score']

# 5. 非线性变换
print("构建非线性变换特征...")
df_full['log_duration'] = np.log1p(df['duration_min'])
df_full['log_distance'] = np.log1p(df['distance_km'])
df_full['speed_distance'] = df['speed_kmh'] * df['distance_km']
df_full['distance_per_duration'] = df['distance_km'] / (df['duration_min'] + 0.1)

transform_features = ['log_duration', 'log_distance', 'speed_distance', 'distance_per_duration']

# 整合所有特征
all_features = (base_features + freq_features + domain_features +
                time_features + transform_features)

print(f"\n总特征数: {len(all_features)}")

# ==================== 准备数据 ====================
X = df_full[all_features].fillna(df_full[all_features].median())
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

# ==================== 精细网格搜索 ====================
print("\n开始精细网格搜索...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid = {
    'C': [0.005, 0.008, 0.01, 0.012, 0.015, 0.02],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': ['balanced'],
    'max_iter': [50000]
}

grid = GridSearchCV(
    LinearSVC(random_state=42, dual='auto'),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳CV F1: {grid.best_score_:.4f}")

# ==================== 测试集评估 ====================
y_pred = grid.predict(X_test_scaled)
f1_test = f1_score(y_test, y_pred, average='weighted')
acc_test = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("📊 最终结果")
print("="*70)
print(f"特征数: {len(all_features)}")
print(f"最佳参数: {grid.best_params_}")
print(f"CV F1:   {grid.best_score_:.4f}")
print(f"Test F1: {f1_test:.4f}")
print(f"Test Acc: {acc_test:.4f}")

print("\n历史对比:")
print(f"  8特征+频率:    F1 = 0.6271")
print(f"  完整37特征:    F1 = 0.6587")
print(f"  当前整合方案:   F1 = {f1_test:.4f}")

if f1_test >= 0.70:
    print(f"\n🎉🎉🎉 成功达到目标！F1 = {f1_test:.4f} >= 0.70")
elif f1_test >= 0.68:
    print(f"\n🎊 非常接近！距目标仅差 {0.70 - f1_test:.4f}")
elif f1_test >= 0.65:
    print(f"\n✅ 显著改进！距目标还差 {0.70 - f1_test:.4f}")
else:
    print(f"\n继续探索...")

# 详细分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['casual', 'member']))

# 保存结果
results = pd.DataFrame([{
    'method': 'Integrated Best (All Innovations)',
    'n_features': len(all_features),
    'cv_f1': grid.best_score_,
    'test_f1': f1_test,
    'test_acc': acc_test,
    'best_params': str(grid.best_params_)
}])

results.to_csv('outputs/tables/integrated_best_results.csv', index=False)
print(f"\n结果已保存: outputs/tables/integrated_best_results.csv")

# 如果效果好，保存模型
if f1_test > 0.65:
    import joblib
    joblib.dump(grid.best_estimator_, 'outputs/models/integrated_best_model.joblib')
    print(f"模型已保存: outputs/models/integrated_best_model.joblib")
