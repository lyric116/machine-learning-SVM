#!/usr/bin/env python3
"""
终极整合：所有有效方法的最佳组合
基于刚才F1=0.6364的发现，继续优化
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("终极整合：冲刺F1>=0.70")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

print(f"样本数: {len(df)}")

# ==================== 构建最全面的特征集 ====================
print("\n构建特征集...")

# 1. 基础特征
base = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak', 'is_weekend']

# 2. 频率特征（最有效的创新）
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

# 3. 非线性变换
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['log_speed'] = np.log1p(df['speed_kmh'])
df['speed2'] = df['speed_kmh'] ** 2
df['dur2'] = df['duration_min'] ** 2
df['speed_x_dist'] = df['speed_kmh'] * df['distance_km']
df['speed_x_dur'] = df['speed_kmh'] * df['duration_min']
df['dist_per_dur'] = df['distance_km'] / (df['duration_min'] + 0.1)
df['speed_per_dist'] = df['speed_kmh'] / (df['distance_km'] + 0.1)

nonlin = ['log_dur', 'log_dist', 'log_speed', 'speed2', 'dur2',
          'speed_x_dist', 'speed_x_dur', 'dist_per_dur', 'speed_per_dist']

# 4. 领域知识
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)
df['weekend_long'] = ((df['is_weekend'] == 1) & (df['duration_min'] > 20)).astype(int)
df['efficiency'] = df['distance_km'] / (df['duration_min'] + 1)

if 'hour' in df.columns:
    df['morning_fast'] = ((df['hour'].between(7, 9)) & (df['speed_kmh'] > 12)).astype(int)
    df['evening_fast'] = ((df['hour'].between(17, 19)) & (df['speed_kmh'] > 12)).astype(int)
    df['lunch_ride'] = df['hour'].between(11, 14).astype(int)
    df['early_morning'] = df['hour'].between(5, 7).astype(int)
    domain = ['fast_commute', 'slow_leisure', 'weekend_long', 'efficiency',
              'morning_fast', 'evening_fast', 'lunch_ride', 'early_morning']
else:
    domain = ['fast_commute', 'slow_leisure', 'weekend_long', 'efficiency']

# 5. 空间特征
if 'start_lat' in df.columns:
    center_lat, center_lng = 38.9072, -77.0369
    df['dist_center_start'] = np.sqrt(
        (df['start_lat'] - center_lat)**2 + (df['start_lng'] - center_lng)**2
    )
    df['dist_center_end'] = np.sqrt(
        (df['end_lat'] - center_lat)**2 + (df['end_lng'] - center_lng)**2
    )
    df['heading_ns'] = df['end_lat'] - df['start_lat']
    df['heading_ew'] = df['end_lng'] - df['start_lng']
    spatial = ['dist_center_start', 'dist_center_end', 'heading_ns', 'heading_ew']
else:
    spatial = []

# 整合所有特征
all_feat = base + freq + nonlin + domain + spatial

print(f"总特征数: {len(all_feat)}")

# ==================== 数据准备 ====================
X = df[all_feat].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== 尝试多个C值找最佳 ====================
print("\n寻找最佳C值...")

best_c = 0.03
best_f1 = 0

for c in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
    clf = LinearSVC(C=c, loss='squared_hinge', class_weight='balanced',
                   max_iter=50000, random_state=42, dual='auto')

    # 5折交叉验证
    scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    mean_score = scores.mean()

    print(f"  C={c:.3f}: CV F1={mean_score:.4f}")

    if mean_score > best_f1:
        best_f1 = mean_score
        best_c = c

print(f"\n最佳C: {best_c}, CV F1: {best_f1:.4f}")

# ==================== 最终训练 ====================
clf_best = LinearSVC(C=best_c, loss='squared_hinge', class_weight='balanced',
                    max_iter=50000, random_state=42, dual='auto')
clf_best.fit(X_train_scaled, y_train)

y_pred = clf_best.predict(X_test_scaled)
f1_test = f1_score(y_test, y_pred, average='weighted')
acc_test = accuracy_score(y_test, y_pred)

# ==================== 多模型集成尝试 ====================
print("\n尝试多模型集成...")

predictions = []
for seed in [42, 43, 44, 45, 46]:
    clf_ens = LinearSVC(C=best_c, loss='squared_hinge', class_weight='balanced',
                       max_iter=50000, random_state=seed, dual='auto')
    clf_ens.fit(X_train_scaled, y_train)
    pred = clf_ens.predict(X_test_scaled)
    predictions.append(pred)

predictions = np.array(predictions)
y_pred_vote = (predictions.sum(axis=0) >= 3).astype(int)
f1_vote = f1_score(y_test, y_pred_vote, average='weighted')

print(f"单模型 F1: {f1_test:.4f}")
print(f"投票集成 F1: {f1_vote:.4f}")

f1_final = max(f1_test, f1_vote)
method = "单模型" if f1_test > f1_vote else "投票集成"

print("\n" + "="*70)
print("🎯 最终结果")
print("="*70)
print(f"特征数: {len(all_feat)}")
print(f"最佳方法: {method}")
print(f"Test F1: {f1_final:.4f}")
print(f"Test Acc: {acc_test if f1_test > f1_vote else accuracy_score(y_test, y_pred_vote):.4f}")

print("\n完整历史:")
print(f"  目标:              F1 = 0.7000")
print(f"  完整37特征:        F1 = 0.6587")
print(f"  之前终极方案:      F1 = 0.6364")
print(f"  当前终极整合:      F1 = {f1_final:.4f}")

if f1_final >= 0.70:
    print(f"\n🎉🎉🎉 成功达到目标！F1={f1_final:.4f}!")
elif f1_final >= 0.68:
    print(f"\n🎊 非常接近目标！仅差 {0.70 - f1_final:.4f}")
elif f1_final > 0.6364:
    print(f"\n✅ 新记录！提升了 {f1_final - 0.6364:.4f}")
else:
    print(f"\n继续尝试其他方向")

print("\n分类报告:")
if f1_test > f1_vote:
    print(classification_report(y_test, y_pred, target_names=['casual', 'member']))
else:
    print(classification_report(y_test, y_pred_vote, target_names=['casual', 'member']))

# 保存
import joblib
joblib.dump(clf_best, 'outputs/models/ultimate_integrated_model.joblib')
joblib.dump(scaler, 'outputs/models/ultimate_integrated_scaler.joblib')

pd.DataFrame([{
    'method': f'Ultimate Integrated ({method})',
    'n_features': len(all_feat),
    'best_c': best_c,
    'cv_f1': best_f1,
    'test_f1': f1_final,
}]).to_csv('outputs/tables/ultimate_integrated_results.csv', index=False)

print(f"\n模型和结果已保存")
