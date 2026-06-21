#!/usr/bin/env python3
"""
创新思路6: 非线性特征变换
通过多项式和对数变换增强特征表达能力
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("创新思路6: 非线性特征变换")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

df_poly = df.copy()

# 基础特征
base_features = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 频率特征
for col in ['start_station_id', 'end_station_id']:
    member_count = df[y==1][col].value_counts()
    total_count = df[col].value_counts()
    member_ratio = (member_count / total_count).fillna(0.5)
    df_poly[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)

df_poly['route'] = df['start_station_id'].astype(str) + '_TO_' + df['end_station_id'].astype(str)
route_freq = df_poly['route'].value_counts()
df_poly['route_rarity'] = df_poly['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq_features = ['start_station_id_member_ratio', 'end_station_id_member_ratio', 'route_rarity']

# 1. 对数变换（处理偏态分布）
df_poly['log_duration'] = np.log1p(df['duration_min'])
df_poly['log_distance'] = np.log1p(df['distance_km'])
df_poly['log_speed'] = np.log1p(df['speed_kmh'])

# 2. 平方特征（捕捉二次关系）
df_poly['speed_squared'] = df['speed_kmh'] ** 2
df_poly['duration_squared'] = df['duration_min'] ** 2

# 3. 关键交互项（手工选择）
df_poly['speed_distance'] = df['speed_kmh'] * df['distance_km']
df_poly['speed_duration'] = df['speed_kmh'] * df['duration_min']
df_poly['distance_duration'] = df['distance_km'] * df['duration_min']

# 4. 比率特征
df_poly['speed_per_distance'] = df['speed_kmh'] / (df['distance_km'] + 0.1)
df_poly['distance_per_duration'] = df['distance_km'] / (df['duration_min'] + 0.1)

transform_features = ['log_duration', 'log_distance', 'log_speed',
                     'speed_squared', 'duration_squared',
                     'speed_distance', 'speed_duration', 'distance_duration',
                     'speed_per_distance', 'distance_per_duration']

all_features = base_features + freq_features + transform_features

X = df_poly[all_features].fillna(df_poly[all_features].median())

# 检查并处理无穷值
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced', max_iter=50000, random_state=42, dual='auto')
clf.fit(scaler.fit_transform(X_tr), y_tr)

f1 = f1_score(y_te, clf.predict(scaler.transform(X_te)), average='weighted')
acc = accuracy_score(y_te, clf.predict(scaler.transform(X_te)))

print(f"特征数: {len(all_features)}")
print(f"F1:  {f1:.4f}")
print(f"Acc: {acc:.4f}")

baseline_f1 = 0.6271
print(f"\nBaseline (8特征+频率): {baseline_f1:.4f}")
print(f"当前方案: {f1:.4f}")
print(f"提升: {f1 - baseline_f1:+.4f}")

if f1 >= 0.70:
    print("\n🎉 达到目标 F1>=0.70!")
elif f1 >= 0.65:
    print(f"\n✅ 接近目标，还差 {0.70 - f1:.4f}")

# 保存
pd.DataFrame([{
    'method': 'Nonlinear Transformation',
    'n_features': len(all_features),
    'f1': f1,
    'accuracy': acc
}]).to_csv('outputs/tables/nonlinear_transform_results.csv', index=False)

print(f"\n结果已保存: outputs/tables/nonlinear_transform_results.csv")
