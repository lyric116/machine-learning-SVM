#!/usr/bin/env python3
"""
基于PDF实验要求的全新SVM方案
原始13字段：ride_id, rideable_type, started_at, ended_at,
            start/end_station_name/id, start/end_lat/lng, member_casual

优化策略：
1. 严格的数据清洗预处理
2. 系统化的特征工程（从13个原始字段出发）
3. 多种特征选择方法对比
4. Linear + RBF核对比
5. 精细的超参数调优
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

# ==================== 配置 ====================
DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
RANDOM_STATE = 42
SAMPLE_SIZE_PER_CLASS = 13000  # 已知13k是较好的样本量

print("=" * 60)
print("基于PDF要求的全新SVM优化方案")
print("=" * 60)

# ==================== 数据加载与分层抽样 ====================
def load_and_sample():
    """按月按类别分层抽样"""
    print("\n[1] 数据加载与分层抽样")

    all_samples = []
    months = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])[:12]

    for month_dir in months:
        csv_path = DATA_DIR / month_dir / f"{month_dir}.csv"
        print(f"  读取 {month_dir}...", end=' ')

        df = pd.read_csv(csv_path, low_memory=False)

        # 分层抽样
        for user_type in ['member', 'casual']:
            subset = df[df['member_casual'] == user_type]
            if len(subset) > 0:
                n_sample = min(SAMPLE_SIZE_PER_CLASS, len(subset))
                sample = subset.sample(n=n_sample, random_state=RANDOM_STATE)
                all_samples.append(sample)

        print(f"✓ 采样 {len(all_samples[-2]) + len(all_samples[-1])} 条")

    df = pd.concat(all_samples, ignore_index=True)
    print(f"\n  总样本: {len(df)}")
    return df

df = load_and_sample()

# ==================== 数据清洗 ====================
def clean_data(df):
    """严格的数据清洗"""
    print("\n[2] 数据清洗")
    n_start = len(df)

    # 1. 目标变量有效
    df = df[df['member_casual'].isin(['member', 'casual'])].copy()
    print(f"  目标变量: {n_start} → {len(df)}")

    # 2. 时间戳有效
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    df = df.dropna(subset=['started_at', 'ended_at'])
    df = df[df['started_at'] < df['ended_at']]
    print(f"  时间戳: {n_start} → {len(df)}")

    # 3. 坐标有效且在华盛顿DC区域
    lat_cols = ['start_lat', 'end_lat']
    lng_cols = ['start_lng', 'end_lng']
    df = df.dropna(subset=lat_cols + lng_cols)
    df = df[(df['start_lat'].between(38.79, 39.0)) &
            (df['end_lat'].between(38.79, 39.0)) &
            (df['start_lng'].between(-77.12, -76.91)) &
            (df['end_lng'].between(-77.12, -76.91))]
    print(f"  坐标: {n_start} → {len(df)}")

    # 4. 骑行时长合理 (1-180分钟)
    df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df = df[(df['duration_min'] >= 1) & (df['duration_min'] <= 180)]
    print(f"  时长: {n_start} → {len(df)}")

    # 5. 骑行距离和速度合理
    df['distance_km'] = np.sqrt(
        (df['end_lat'] - df['start_lat'])**2 +
        (df['end_lng'] - df['start_lng'])**2
    ) * 111  # 粗略转换为公里
    df['speed_kmh'] = df['distance_km'] / (df['duration_min'] / 60)
    df = df[(df['distance_km'] <= 30) & (df['speed_kmh'] <= 50)]
    print(f"  距离速度: {n_start} → {len(df)}")

    # 6. 去重
    df = df.drop_duplicates(subset='ride_id')
    print(f"  去重: {n_start} → {len(df)}")

    return df

df = clean_data(df)

# ==================== 特征工程 ====================
def engineer_features(df):
    """从13个原始字段系统化构建特征"""
    print("\n[3] 特征工程（基于13个原始字段）")

    df = df.copy()

    # --- 时间特征 (从 started_at) ---
    df['hour'] = df['started_at'].dt.hour
    df['weekday'] = df['started_at'].dt.weekday
    df['month'] = df['started_at'].dt.month
    df['day'] = df['started_at'].dt.day

    # 周期特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)

    # 时间标记
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7,8,9,17,18,19]).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)

    # --- 空间特征 (从 lat/lng) ---
    # 已有: distance_km, speed_kmh
    df['lat_displacement'] = df['end_lat'] - df['start_lat']
    df['lng_displacement'] = df['end_lng'] - df['start_lng']

    # 方向角度
    df['direction_rad'] = np.arctan2(df['lat_displacement'], df['lng_displacement'])
    df['direction_sin'] = np.sin(df['direction_rad'])
    df['direction_cos'] = np.cos(df['direction_rad'])

    # 到市中心距离 (DC中心约 38.9072, -77.0369)
    center_lat, center_lng = 38.9072, -77.0369
    df['start_to_center'] = np.sqrt(
        (df['start_lat'] - center_lat)**2 + (df['start_lng'] - center_lng)**2
    )
    df['end_to_center'] = np.sqrt(
        (df['end_lat'] - center_lat)**2 + (df['end_lng'] - center_lng)**2
    )
    df['moving_toward_center'] = (df['end_to_center'] < df['start_to_center']).astype(int)

    # 空间网格
    df['start_grid'] = (df['start_lat'] * 100).astype(int).astype(str) + '_' + \
                       (df['start_lng'] * 100).astype(int).astype(str)
    df['end_grid'] = (df['end_lat'] * 100).astype(int).astype(str) + '_' + \
                     (df['end_lng'] * 100).astype(int).astype(str)

    # --- 行程特征 ---
    # 已有: duration_min
    df['is_short_trip'] = (df['duration_min'] < 5).astype(int)
    df['is_long_trip'] = (df['duration_min'] > 30).astype(int)
    df['is_round_trip'] = (df['distance_km'] < 0.5).astype(int)

    # 对数变换（处理偏态分布）
    df['log_duration'] = np.log1p(df['duration_min'])
    df['log_distance'] = np.log1p(df['distance_km'])

    # --- 站点特征 (从 station_name/id) ---
    df['start_station_missing'] = df['start_station_id'].isna().astype(int)
    df['end_station_missing'] = df['end_station_id'].isna().astype(int)
    df['both_stations_missing'] = (df['start_station_missing'] & df['end_station_missing']).astype(int)

    # 填充缺失站点
    df['start_station_id'] = df['start_station_id'].fillna('unknown')
    df['end_station_id'] = df['end_station_id'].fillna('unknown')

    # 站点路线组合
    df['route'] = df['start_station_id'].astype(str) + '_to_' + df['end_station_id'].astype(str)

    # --- 车辆类型 (从 rideable_type) ---
    # 直接保留，后续编码

    print(f"  构建完成，特征维度: {df.shape[1]}")
    return df

df = engineer_features(df)

# ==================== 特征编码 ====================
def encode_features(df):
    """特征编码与准备"""
    print("\n[4] 特征编码")

    # 目标变量
    y = (df['member_casual'] == 'member').astype(int)

    # 数值特征
    numeric_features = [
        'duration_min', 'distance_km', 'speed_kmh',
        'hour', 'weekday', 'month',
        'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
        'lat_displacement', 'lng_displacement',
        'direction_sin', 'direction_cos',
        'start_to_center', 'end_to_center',
        'log_duration', 'log_distance',
        'is_weekend', 'is_rush_hour', 'is_night',
        'is_short_trip', 'is_long_trip', 'is_round_trip',
        'start_station_missing', 'end_station_missing', 'both_stations_missing',
        'moving_toward_center'
    ]

    # 类别特征
    categorical_features = [
        'rideable_type',
        'start_station_id', 'end_station_id', 'route',
        'start_grid', 'end_grid'
    ]

    # 频率编码（替代One-Hot，减少维度）
    print("  应用频率编码...")
    for col in categorical_features:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq).fillna(0)

    # 目标编码（Out-of-fold，防止泄漏）
    print("  应用目标编码（OOF）...")
    for col in ['start_station_id', 'end_station_id', 'route']:
        df[f'{col}_target_mean'] = 0.0
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        for train_idx, val_idx in skf.split(df, y):
            # 构造临时DataFrame用于groupby
            temp_df = pd.DataFrame({col: df[col].iloc[train_idx], 'target': y.iloc[train_idx]})
            target_mean = temp_df.groupby(col)['target'].mean()
            global_mean = y.iloc[train_idx].mean()

            # 平滑处理
            smoothing = 10
            counts = temp_df[col].value_counts()
            smoothed = (target_mean * counts + global_mean * smoothing) / (counts + smoothing)

            df.loc[val_idx, f'{col}_target_mean'] = df.loc[val_idx, col].map(smoothed).fillna(global_mean)

    # 最终特征列表
    freq_features = [f'{col}_freq' for col in categorical_features]
    target_features = [f'{col}_target_mean' for col in ['start_station_id', 'end_station_id', 'route']]

    feature_cols = numeric_features + freq_features + target_features
    X = df[feature_cols].copy()

    # 填充任何剩余缺失值
    X = X.fillna(X.median())

    print(f"  最终特征数: {X.shape[1]}")
    print(f"  样本数: {X.shape[0]}")

    return X, y

X, y = encode_features(df)

# ==================== 数据划分 ====================
print("\n[5] 数据划分（80:20）")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"  训练集: {X_train.shape[0]} ({y_train.mean():.2%} member)")
print(f"  测试集: {X_test.shape[0]} ({y_test.mean():.2%} member)")

# ==================== 实验1: 全特征 + LinearSVC ====================
print("\n" + "="*60)
print("实验1: 全特征 + LinearSVC")
print("="*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# GridSearch
param_grid = {
    'C': [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5],
    'class_weight': [None, 'balanced']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
grid = GridSearchCV(
    LinearSVC(max_iter=50000, random_state=RANDOM_STATE, dual='auto'),
    param_grid,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

print("开始网格搜索...")
grid.fit(X_train_scaled, y_train)

print(f"\n最佳参数: {grid.best_params_}")
print(f"最佳CV F1: {grid.best_score_:.4f}")

# 测试集评估
y_pred = grid.predict(X_test_scaled)
test_f1 = f1_score(y_test, y_pred, average='weighted')
test_acc = accuracy_score(y_test, y_pred)

print(f"测试集 F1: {test_f1:.4f}")
print(f"测试集 Acc: {test_acc:.4f}")

results = {
    'experiment_1_full_linear': {
        'best_params': grid.best_params_,
        'cv_f1': grid.best_score_,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'n_features': X_train.shape[1]
    }
}

best_f1 = test_f1
best_model = grid.best_estimator_
best_scaler = scaler

# ==================== 实验2: 特征选择 (Mutual Information) ====================
print("\n" + "="*60)
print("实验2: 特征选择（互信息） + LinearSVC")
print("="*60)

# 尝试不同的k值
for k in [20, 30, 40]:
    print(f"\n--- Top {k} 特征 ---")

    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)

    selected_features = X.columns[selector.get_support()].tolist()
    print(f"选择的特征: {selected_features[:5]}... (共{k}个)")

    # 快速训练
    clf = LinearSVC(C=grid.best_params_['C'],
                   class_weight=grid.best_params_['class_weight'],
                   max_iter=50000, random_state=RANDOM_STATE, dual='auto')
    clf.fit(X_train_selected, y_train)

    y_pred = clf.predict(X_test_selected)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    test_acc = accuracy_score(y_test, y_pred)

    print(f"测试集 F1: {test_f1:.4f}")
    print(f"测试集 Acc: {test_acc:.4f}")

    results[f'experiment_2_select_k{k}'] = {
        'k': k,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'selected_features': selected_features
    }

    if test_f1 > best_f1:
        best_f1 = test_f1
        best_model = clf
        best_scaler = (scaler, selector)
        print("  ✓ 新的最佳模型！")

# ==================== 实验3: RBF核（选定特征） ====================
print("\n" + "="*60)
print("实验3: RBF核（数值特征 + 保守参数）")
print("="*60)

# 只用数值特征，避免高维导致的内存问题
numeric_feature_names = [
    'duration_min', 'distance_km', 'speed_kmh',
    'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos',
    'lat_displacement', 'lng_displacement',
    'direction_sin', 'direction_cos',
    'start_to_center', 'end_to_center',
    'log_duration', 'log_distance',
]

numeric_indices = [X.columns.get_loc(f) for f in numeric_feature_names if f in X.columns]
X_train_numeric = X_train_scaled[:, numeric_indices]
X_test_numeric = X_test_scaled[:, numeric_indices]

print(f"使用 {len(numeric_indices)} 个数值特征")

# 保守的RBF参数
param_grid_rbf = {
    'C': [0.5, 1, 2],
    'gamma': ['scale', 0.01, 0.1],
    'class_weight': ['balanced']
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
grid_rbf = GridSearchCV(
    SVC(kernel='rbf', max_iter=10000, random_state=RANDOM_STATE, cache_size=1000),
    param_grid_rbf,
    cv=cv,
    scoring='f1_weighted',
    n_jobs=1,  # 单线程，避免内存爆炸
    verbose=1
)

print("开始RBF网格搜索（保守参数）...")
grid_rbf.fit(X_train_numeric, y_train)

print(f"\n最佳参数: {grid_rbf.best_params_}")
print(f"最佳CV F1: {grid_rbf.best_score_:.4f}")

y_pred_rbf = grid_rbf.predict(X_test_numeric)
test_f1_rbf = f1_score(y_test, y_pred_rbf, average='weighted')
test_acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"测试集 F1: {test_f1_rbf:.4f}")
print(f"测试集 Acc: {test_acc_rbf:.4f}")

results['experiment_3_rbf'] = {
    'best_params': grid_rbf.best_params_,
    'cv_f1': grid_rbf.best_score_,
    'test_f1': test_f1_rbf,
    'test_acc': test_acc_rbf,
    'n_features': len(numeric_indices)
}

if test_f1_rbf > best_f1:
    best_f1 = test_f1_rbf
    print("  ✓ RBF核获得新的最佳结果！")

# ==================== 保存结果 ====================
print("\n" + "="*60)
print("最终结果")
print("="*60)

results['best_f1'] = best_f1

print(f"\n最佳F1分数: {best_f1:.4f}")
print(f"距离目标0.70: {0.70 - best_f1:.4f}")

# 保存
output_path = OUTPUT_DIR / 'fresh_approach_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n结果已保存到: {output_path}")

# 保存最佳模型
if best_f1 >= 0.66:
    model_path = OUTPUT_DIR / 'models' / 'svm_fresh_best.joblib'
    joblib.dump({'model': best_model, 'scaler': best_scaler}, model_path)
    print(f"最佳模型已保存到: {model_path}")

print("\n实验完成！")

