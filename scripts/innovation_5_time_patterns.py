#!/usr/bin/env python3
"""
创新思路5: 基于时间序列的用户行为建模
假设：骑行时间模式本身就是区分特征
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
print("创新思路5: 时间模式深度建模")
print("="*70)

df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

# 构建时间模式特征
df_time = df.copy()

if 'hour' in df.columns:
    # 1. 多时段标记（更细粒度）
    df_time['early_morning'] = df['hour'].between(5, 7).astype(int)   # 早晨
    df_time['morning_rush'] = df['hour'].between(7, 9).astype(int)    # 早高峰
    df_time['midday'] = df['hour'].between(10, 16).astype(int)        # 白天
    df_time['evening_rush'] = df['hour'].between(17, 19).astype(int)  # 晚高峰
    df_time['night'] = df['hour'].between(20, 23).astype(int)         # 夜晚

    # 2. 时段×速度特征
    df_time['rush_speed_score'] = (
        df_time['morning_rush'] * (df['speed_kmh'] > 12).astype(int) +
        df_time['evening_rush'] * (df['speed_kmh'] > 12).astype(int)
    )

    # 3. 时段×时长特征
    df_time['midday_long'] = (df_time['midday'] * (df['duration_min'] > 20)).astype(int)

# 4. 星期×时段交互
if 'weekday' in df.columns and 'hour' in df.columns:
    # 工作日早晚高峰
    df_time['weekday_rush'] = (
        (df['weekday'] < 5) &
        ((df['hour'].between(7, 9)) | (df['hour'].between(17, 19)))
    ).astype(int)

    # 周末白天
    df_time['weekend_daytime'] = (
        (df['weekday'] >= 5) &
        (df['hour'].between(10, 17))
    ).astype(int)

# 5. 速度稳定性（假设会员速度更稳定）
# 使用速度与中位速度的偏差
median_speed = df['speed_kmh'].median()
df_time['speed_deviation'] = np.abs(df['speed_kmh'] - median_speed)

# 基础+频率+时间模式
base_features = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 构建频率特征
for col in ['start_station_id', 'end_station_id']:
    if col in df.columns:
        member_count = df[y==1][col].value_counts()
        total_count = df[col].value_counts()
        member_ratio = (member_count / total_count).fillna(0.5)
        df_time[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)

df_time['route'] = df['start_station_id'].astype(str) + '_TO_' + df['end_station_id'].astype(str)
route_freq = df_time['route'].value_counts()
df_time['route_rarity'] = df_time['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq_features = ['start_station_id_member_ratio', 'end_station_id_member_ratio', 'route_rarity']

time_features = ['early_morning', 'morning_rush', 'midday', 'evening_rush', 'night',
                'rush_speed_score', 'midday_long', 'weekday_rush', 'weekend_daytime',
                'speed_deviation']

all_features = base_features + freq_features + time_features

X = df_time[all_features].fillna(df_time[all_features].median())
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
else:
    print(f"\n需要继续优化")

# 保存
pd.DataFrame([{
    'method': 'Time Pattern Modeling',
    'n_features': len(all_features),
    'f1': f1,
    'accuracy': acc
}]).to_csv('outputs/tables/time_pattern_results.csv', index=False)

print(f"\n结果已保存: outputs/tables/time_pattern_results.csv")
