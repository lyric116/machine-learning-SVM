#!/usr/bin/env python3
"""
激进方案：从原始数据重新采样更大规模
目标：突破F1=0.70
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("激进方案：大规模采样 + 最优特征")
print("="*70)

# ==================== 从原始数据重新采样 ====================
print("\n步骤1: 从原始数据重新采样（目标：15k-20k/class/month）")

data_dir = Path('data')
months = [f'2025{m:02d}' for m in range(1, 13)]

samples = []
target_per_month_per_class = 15000  # 增加到15k

for month in months:
    month_dir = data_dir / month
    if not month_dir.exists():
        continue

    csv_files = list(month_dir.glob('*.csv'))
    if not csv_files:
        continue

    print(f"  处理 {month}...")

    # 读取该月所有数据
    month_data = []
    for csv_file in csv_files:
        try:
            chunk = pd.read_csv(csv_file, low_memory=False)
            month_data.append(chunk)
        except:
            continue

    if not month_data:
        continue

    df_month = pd.concat(month_data, ignore_index=True)

    # 按类别采样
    for user_type in ['member', 'casual']:
        df_type = df_month[df_month['member_casual'] == user_type]

        if len(df_type) > 0:
            n_sample = min(target_per_month_per_class, len(df_type))
            sample = df_type.sample(n=n_sample, random_state=42)
            samples.append(sample)
            print(f"    {user_type}: {n_sample} 样本")

if not samples:
    print("⚠️ 无法读取原始数据，使用已清洗样本")
    df = pd.read_csv('outputs/processed/cleaned_stratified_sample.csv')
else:
    df_sampled = pd.concat(samples, ignore_index=True)
    print(f"\n总采样: {len(df_sampled)}")

    # 快速清洗
    print("\n步骤2: 快速数据清洗...")

    # 删除缺失目标
    df = df_sampled[df_sampled['member_casual'].isin(['member', 'casual'])].copy()
    print(f"  有效目标: {len(df)}")

    # 删除时间异常
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    df = df.dropna(subset=['started_at', 'ended_at'])
    df = df[df['ended_at'] > df['started_at']]
    print(f"  有效时间: {len(df)}")

    # 计算基础特征
    df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60

    # 删除异常时长
    df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
    print(f"  有效时长: {len(df)}")

    # 坐标有效性
    coord_cols = ['start_lat', 'start_lng', 'end_lat', 'end_lng']
    df = df.dropna(subset=coord_cols)

    # DC区域范围
    df = df[
        (df['start_lat'].between(38.79, 39.0)) &
        (df['start_lng'].between(-77.12, -76.91)) &
        (df['end_lat'].between(38.79, 39.0)) &
        (df['end_lng'].between(-76.91, -77.12))
    ]
    print(f"  有效坐标: {len(df)}")

    # 计算距离和速度
    df['distance_km'] = np.sqrt(
        ((df['end_lat'] - df['start_lat']) * 111)**2 +
        ((df['end_lng'] - df['start_lng']) * 85)**2
    )
    df['speed_kmh'] = df['distance_km'] / (df['duration_min'] / 60)

    # 删除异常速度
    df = df[(df['speed_kmh'] >= 2) & (df['speed_kmh'] <= 30)]
    print(f"  有效速度: {len(df)}")

    # 提取时间特征
    df['hour'] = df['started_at'].dt.hour
    df['weekday'] = df['started_at'].dt.weekday
    df['month'] = df['started_at'].dt.month
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_commute_peak'] = (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 17) & (df['hour'] <= 19))
    ).astype(int)

    print(f"\n✓ 清洗完成，最终样本: {len(df)}")

# ==================== 构建最优特征集 ====================
print("\n步骤3: 构建最优特征集...")

y = (df['member_casual'] == 'member').astype(int)

# 基础
base = ['speed_kmh', 'weekday', 'distance_km', 'duration_min', 'is_commute_peak']

# 频率
for col in ['start_station_id', 'end_station_id']:
    if col in df.columns:
        member_cnt = df[y==1][col].value_counts()
        total_cnt = df[col].value_counts()
        ratio = (member_cnt / total_cnt).fillna(0.5)
        df[f'{col}_mr'] = df[col].map(ratio).fillna(0.5)

if 'start_station_id' in df.columns and 'end_station_id' in df.columns:
    df['route'] = df['start_station_id'].astype(str) + '_' + df['end_station_id'].astype(str)
    route_freq = df['route'].value_counts()
    df['route_rare'] = df['route'].map(route_freq).apply(lambda x: 1/(x+1))

freq = ['start_station_id_mr', 'end_station_id_mr', 'route_rare']

# 非线性
df['log_dur'] = np.log1p(df['duration_min'])
df['log_dist'] = np.log1p(df['distance_km'])
df['speed_dist'] = df['speed_kmh'] * df['distance_km']
df['dist_per_dur'] = df['distance_km'] / (df['duration_min'] + 0.1)

nonlin = ['log_dur', 'log_dist', 'speed_dist', 'dist_per_dur']

# 领域
df['fast_commute'] = ((df['speed_kmh'] > 12) & (df['is_commute_peak'] == 1)).astype(int)
df['slow_leisure'] = ((df['speed_kmh'] < 10) & (df['is_weekend'] == 1)).astype(int)

domain = ['fast_commute', 'slow_leisure']

# 空间
if 'start_lat' in df.columns:
    center_lat, center_lng = 38.9072, -77.0369
    df['dist_center'] = np.sqrt(
        (df['start_lat'] - center_lat)**2 + (df['start_lng'] - center_lng)**2
    )
    spatial = ['dist_center']
else:
    spatial = []

all_feat = base + freq + nonlin + domain + spatial

print(f"总特征数: {len(all_feat)}")

# ==================== 训练 ====================
X = df[all_feat].copy()
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n步骤4: 训练模型")
print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 使用之前找到的最佳参数
clf = LinearSVC(
    C=0.012,
    loss='squared_hinge',
    class_weight='balanced',
    max_iter=50000,
    random_state=42,
    dual='auto'
)

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred, average='weighted')
acc = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("📊 最终结果")
print("="*70)
print(f"样本量: {len(df)}")
print(f"特征数: {len(all_feat)}")
print(f"Test F1: {f1:.4f}")
print(f"Test Acc: {acc:.4f}")

print("\n对比:")
print(f"  目标:          F1 = 0.7000")
print(f"  当前最佳:      F1 = 0.6308")
print(f"  本次尝试:      F1 = {f1:.4f}")

if f1 >= 0.70:
    print(f"\n🎉🎉🎉 成功达到目标！")
elif f1 > 0.6308:
    print(f"\n🎊 新记录！提升 {f1 - 0.6308:.4f}")
elif f1 > 0.65:
    print(f"\n✅ 有改进")
else:
    print(f"\n继续尝试")

# 保存
pd.DataFrame([{
    'method': 'Large Sample Resample',
    'n_samples': len(df),
    'n_features': len(all_feat),
    'test_f1': f1,
    'test_acc': acc
}]).to_csv('outputs/tables/large_sample_results.csv', index=False)

print(f"\n结果已保存: outputs/tables/large_sample_results.csv")
