#!/usr/bin/env python3
"""
基于频率的高级特征优化
测试各种频率统计特征是否能进一步提升性能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

print("基于频率的高级特征实验")
print("="*70)

# 加载数据
df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
y = (df['member_casual'] == 'member').astype(int)

print(f"原始样本数: {len(df)}")

# ==================== 构建频率特征 ====================

def add_frequency_features(df, y):
    """添加各种基于频率的特征"""
    df = df.copy()

    print("\n构建频率特征...")

    # 1. 站点使用频率（会员 vs 散户的差异）
    print("  1. 站点使用频率差异...")
    for col in ['start_station_id', 'end_station_id']:
        if col in df.columns:
            # 总体频率
            total_freq = df[col].value_counts()

            # 会员频率
            member_freq = df[y == 1][col].value_counts()

            # 散户频率
            casual_freq = df[y == 0][col].value_counts()

            # 计算会员占比
            member_ratio = (member_freq / (member_freq + casual_freq)).fillna(0.5)

            df[f'{col}_member_ratio'] = df[col].map(member_ratio).fillna(0.5)
            df[f'{col}_total_freq'] = df[col].map(total_freq).fillna(0)
            df[f'{col}_popularity'] = np.log1p(df[f'{col}_total_freq'])

    # 2. 时段-站点组合频率
    print("  2. 时段-站点组合频率...")
    if 'hour' in df.columns and 'start_station_id' in df.columns:
        df['hour_station'] = df['hour'].astype(str) + '_' + df['start_station_id'].astype(str)

        hour_station_member = df[y == 1]['hour_station'].value_counts()
        hour_station_casual = df[y == 0]['hour_station'].value_counts()
        hour_station_ratio = (hour_station_member /
                             (hour_station_member + hour_station_casual)).fillna(0.5)

        df['hour_station_member_ratio'] = df['hour_station'].map(hour_station_ratio).fillna(0.5)

    # 3. 车辆类型-时段组合
    print("  3. 车辆类型-时段组合...")
    if 'rideable_type' in df.columns and 'weekday' in df.columns:
        df['vehicle_weekday'] = df['rideable_type'].astype(str) + '_' + df['weekday'].astype(str)

        vw_member = df[y == 1]['vehicle_weekday'].value_counts()
        vw_casual = df[y == 0]['vehicle_weekday'].value_counts()
        vw_ratio = (vw_member / (vw_member + vw_casual)).fillna(0.5)

        df['vehicle_weekday_ratio'] = df['vehicle_weekday'].map(vw_ratio).fillna(0.5)

    # 4. 路线稀有度（罕见路线可能是散户）
    print("  4. 路线稀有度...")
    if 'start_station_id' in df.columns and 'end_station_id' in df.columns:
        df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
        route_freq = df['route'].value_counts()
        df['route_frequency'] = df['route'].map(route_freq).fillna(1)
        df['is_rare_route'] = (df['route_frequency'] < 10).astype(int)
        df['route_rarity_score'] = 1 / (df['route_frequency'] + 1)

    # 5. 速度区间在该时段的频率
    print("  5. 速度-时段组合频率...")
    if 'speed_kmh' in df.columns and 'hour' in df.columns:
        df['speed_bin'] = pd.cut(df['speed_kmh'], bins=[0, 8, 12, 16, 50],
                                 labels=['slow', 'medium', 'fast', 'very_fast'])
        df['speed_hour'] = df['speed_bin'].astype(str) + '_' + df['hour'].astype(str)

        sh_member = df[y == 1]['speed_hour'].value_counts()
        sh_casual = df[y == 0]['speed_hour'].value_counts()
        sh_ratio = (sh_member / (sh_member + sh_casual)).fillna(0.5)

        df['speed_hour_member_ratio'] = df['speed_hour'].map(sh_ratio).fillna(0.5)

    # 6. 时长模式频率
    print("  6. 时长模式频率...")
    if 'duration_min' in df.columns:
        df['duration_category'] = pd.cut(df['duration_min'],
                                        bins=[0, 5, 10, 20, 180],
                                        labels=['very_short', 'short', 'medium', 'long'])

        dur_member = df[y == 1]['duration_category'].value_counts()
        dur_casual = df[y == 0]['duration_category'].value_counts()
        dur_ratio = (dur_member / (dur_member + dur_casual)).fillna(0.5)

        df['duration_pattern_ratio'] = df['duration_category'].map(dur_ratio).fillna(0.5)

    # 7. 工作日vs周末的站点使用差异
    print("  7. 工作日/周末站点差异...")
    if 'is_weekend' in df.columns and 'start_station_id' in df.columns:
        df['weekend_station'] = df['is_weekend'].astype(str) + '_' + df['start_station_id'].astype(str)

        ws_member = df[y == 1]['weekend_station'].value_counts()
        ws_casual = df[y == 0]['weekend_station'].value_counts()
        ws_ratio = (ws_member / (ws_member + ws_casual)).fillna(0.5)

        df['weekend_station_ratio'] = df['weekend_station'].map(ws_ratio).fillna(0.5)

    print(f"  新增特征后总列数: {df.shape[1]}")

    return df

df_enhanced = add_frequency_features(df, y)

# ==================== 选择特征 ====================

# 基础7特征（之前的最佳）
base_features = [
    'speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak'
]

# 新增的频率特征
frequency_features = [
    'start_station_id_member_ratio',
    'end_station_id_member_ratio',
    'start_station_id_popularity',
    'end_station_id_popularity',
    'hour_station_member_ratio',
    'vehicle_weekday_ratio',
    'route_rarity_score',
    'is_rare_route',
    'speed_hour_member_ratio',
    'duration_pattern_ratio',
    'weekend_station_ratio'
]

# ==================== 实验对比 ====================

results = []

# 实验1: 仅基础特征（baseline）
print("\n" + "="*70)
print("实验1: 基础7特征 (baseline)")
print("="*70)

X_base = df_enhanced[base_features].copy()
X_base = X_base.fillna(X_base.median())

X_train, X_test, y_train, y_test = train_test_split(
    X_base, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
               max_iter=50000, random_state=42, dual='auto')
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
f1_base = f1_score(y_test, y_pred, average='weighted')
acc_base = accuracy_score(y_test, y_pred)

print(f"F1:  {f1_base:.4f}")
print(f"Acc: {acc_base:.4f}")

results.append({
    'experiment': 'Baseline (7 features)',
    'n_features': len(base_features),
    'f1': f1_base,
    'accuracy': acc_base
})

# 实验2: 基础 + 所有频率特征
print("\n" + "="*70)
print("实验2: 基础7特征 + 所有频率特征")
print("="*70)

all_freq_features = base_features + frequency_features
X_freq = df_enhanced[all_freq_features].copy()

# 处理类别列
for col in X_freq.columns:
    if X_freq[col].dtype == 'object':
        le = LabelEncoder()
        X_freq[col] = le.fit_transform(X_freq[col].fillna('missing'))
    else:
        X_freq[col] = X_freq[col].fillna(X_freq[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X_freq, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
               max_iter=50000, random_state=42, dual='auto')
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
f1_freq = f1_score(y_test, y_pred, average='weighted')
acc_freq = accuracy_score(y_test, y_pred)

print(f"F1:  {f1_freq:.4f}")
print(f"Acc: {acc_freq:.4f}")
print(f"提升: {f1_freq - f1_base:+.4f}")

results.append({
    'experiment': 'Base + All Frequency (18 features)',
    'n_features': len(all_freq_features),
    'f1': f1_freq,
    'accuracy': acc_freq
})

# 实验3: 精选频率特征（手工选择最有效的）
print("\n" + "="*70)
print("实验3: 基础7特征 + 精选频率特征")
print("="*70)

selected_freq = [
    'start_station_id_member_ratio',
    'end_station_id_member_ratio',
    'hour_station_member_ratio',
    'route_rarity_score',
    'speed_hour_member_ratio'
]

selected_features = base_features + selected_freq
X_selected = df_enhanced[selected_features].copy()

for col in X_selected.columns:
    if X_selected[col].dtype == 'object':
        le = LabelEncoder()
        X_selected[col] = le.fit_transform(X_selected[col].fillna('missing'))
    else:
        X_selected[col] = X_selected[col].fillna(X_selected[col].median())

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = LinearSVC(C=0.01, loss='squared_hinge', class_weight='balanced',
               max_iter=50000, random_state=42, dual='auto')
clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
f1_selected = f1_score(y_test, y_pred, average='weighted')
acc_selected = accuracy_score(y_test, y_pred)

print(f"F1:  {f1_selected:.4f}")
print(f"Acc: {acc_selected:.4f}")
print(f"提升: {f1_selected - f1_base:+.4f}")

results.append({
    'experiment': 'Base + Selected Frequency (12 features)',
    'n_features': len(selected_features),
    'f1': f1_selected,
    'accuracy': acc_selected
})

# ==================== 总结 ====================

print("\n" + "="*70)
print("实验总结")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

print("\n对比完整37特征模型:")
print(f"  完整模型:      F1 = 0.6587")
print(f"  基础7特征:     F1 = {f1_base:.4f} (差距: {0.6587-f1_base:.4f})")
print(f"  +所有频率:     F1 = {f1_freq:.4f} (差距: {0.6587-f1_freq:.4f})")
print(f"  +精选频率:     F1 = {f1_selected:.4f} (差距: {0.6587-f1_selected:.4f})")

# 找到最佳
best_idx = results_df['f1'].idxmax()
best = results_df.iloc[best_idx]

print("\n" + "="*70)
print("💡 结论")
print("="*70)

if best['f1'] > f1_base + 0.005:
    print(f"✅ 频率特征有效！")
    print(f"   {best['experiment']}")
    print(f"   F1提升: {best['f1'] - f1_base:+.4f} ({(best['f1']-f1_base)/f1_base*100:+.2f}%)")

    if best['f1'] > 0.62:
        print(f"\n🎉 突破0.62！距离完整模型仅差 {0.6587 - best['f1']:.4f}")
else:
    print(f"⚠️ 频率特征提升有限")
    print(f"   最大提升: {best['f1'] - f1_base:+.4f}")
    print(f"   可能已接近7特征的性能上限")

# 保存结果
results_df.to_csv('outputs/tables/frequency_features_results.csv', index=False)
print(f"\n结果已保存: outputs/tables/frequency_features_results.csv")
