#!/usr/bin/env python3
"""
Capital Bikeshare SVM分类 - 最终优化版本
基于历史最佳结果 F1=0.6587 的配置
- 样本量: 13k/class/month
- 特征数: 37个精选特征
- 模型: LinearSVC (C=0.003, loss=hinge, class_weight=balanced)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score, precision_score, recall_score
)
import joblib
import json

# ==================== 配置 ====================
DATA_DIR = Path('data')
OUTPUT_DIR = Path('outputs')
MODEL_DIR = OUTPUT_DIR / 'models'
FIGURE_DIR = OUTPUT_DIR / 'figures'
TABLE_DIR = OUTPUT_DIR / 'tables'

RANDOM_STATE = 42
SAMPLE_SIZE = 13000  # 每月每类别采样数（历史最佳）

print("="*70)
print("Capital Bikeshare SVM分类 - 最终优化版本")
print(f"目标: 基于最佳配置 (F1=0.6587) 重现并完善")
print("="*70)

# ==================== 辅助函数 ====================
class TargetMeanEncoder:
    """Out-of-fold目标均值编码器"""
    def __init__(self, smoothing=10, n_folds=5, random_state=42):
        self.smoothing = smoothing
        self.n_folds = n_folds
        self.random_state = random_state
        self.target_mean_map = {}
        self.global_mean = None

    def fit(self, X, y):
        """训练时使用Out-of-fold编码"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        self.target_mean_map = {}
        self.global_mean = y.mean()

        for col in X.columns:
            # 训练集全局映射（用于测试集）
            col_mean = X.groupby(col)[y].mean()
            col_count = X.groupby(col)[y].count()
            smoothed = (col_mean * col_count + self.global_mean * self.smoothing) / \
                      (col_count + self.smoothing)
            self.target_mean_map[col] = smoothed.to_dict()

        return self

    def transform(self, X):
        """转换"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_encoded = pd.DataFrame()

        for col in X.columns:
            X_encoded[f'{col}_tm'] = X[col].map(self.target_mean_map.get(col, {})).fillna(self.global_mean)

        return X_encoded.values

    def fit_transform(self, X, y):
        """使用OOF方式"""
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        X_encoded = pd.DataFrame(index=X.index)

        # 先fit用于后续transform
        self.fit(X, y)

        # OOF编码
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        for col in X.columns:
            X_encoded[f'{col}_tm'] = 0.0

            for train_idx, val_idx in skf.split(X, y):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                col_mean = X_train.groupby(col)[y_train].mean()
                col_count = X_train.groupby(col)[y_train].count()
                global_mean_fold = y_train.mean()

                smoothed = (col_mean * col_count + global_mean_fold * self.smoothing) / \
                          (col_count + self.smoothing)

                X_encoded.loc[X.index[val_idx], f'{col}_tm'] = \
                    X.iloc[val_idx][col].map(smoothed).fillna(global_mean_fold)

        return X_encoded.values

# ==================== 数据加载 ====================
def load_data():
    """加载并分层抽样"""
    print("\n[1] 数据加载与分层抽样")

    all_samples = []
    for month in range(1, 13):
        month_str = f"2025{month:02d}"
        csv_dir = DATA_DIR / f"{month_str}-capitalbikeshare-tripdata"
        csv_file = csv_dir / f"{month_str}-capitalbikeshare-tripdata.csv"

        if not csv_file.exists():
            print(f"  警告: {month_str} 不存在")
            continue

        print(f"  读取 {month_str}...", end=' ')
        df = pd.read_csv(csv_file, low_memory=False)

        for user_type in ['member', 'casual']:
            subset = df[df['member_casual'] == user_type]
            if len(subset) >= SAMPLE_SIZE:
                sample = subset.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)
                all_samples.append(sample)

        print(f"✓")

    df = pd.concat(all_samples, ignore_index=True)
    print(f"  总样本: {len(df)}")
    return df

# ==================== 数据清洗 ====================
def clean_data(df):
    """数据清洗"""
    print("\n[2] 数据清洗")
    n0 = len(df)

    # 时间
    df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
    df['ended_at'] = pd.to_datetime(df['ended_at'], errors='coerce')
    df = df.dropna(subset=['started_at', 'ended_at'])
    df = df[df['started_at'] < df['ended_at']]

    # 坐标
    df = df.dropna(subset=['start_lat', 'end_lat', 'start_lng', 'end_lng'])
    df = df[(df['start_lat'].between(38.79, 39.0)) &
            (df['end_lat'].between(38.79, 39.0)) &
            (df['start_lng'].between(-77.12, -76.91)) &
            (df['end_lng'].between(-77.12, -76.91))]

    # 计算基础指标
    df['duration_min'] = (df['ended_at'] - df['started_at']).dt.total_seconds() / 60
    df['distance_km'] = np.sqrt(
        (df['end_lat'] - df['start_lat'])**2 +
        (df['end_lng'] - df['start_lng'])**2
    ) * 111
    df['speed_kmh'] = df['distance_km'] / (df['duration_min'] / 60 + 1e-6)

    # 过滤异常值
    df = df[(df['duration_min'].between(1, 180)) &
            (df['distance_km'] <= 30) &
            (df['speed_kmh'] <= 50)]

    # 去重
    df = df.drop_duplicates(subset='ride_id')

    print(f"  保留: {len(df)}/{n0} ({len(df)/n0*100:.1f}%)")
    return df

df_raw = load_data()
df_clean = clean_data(df_raw)
