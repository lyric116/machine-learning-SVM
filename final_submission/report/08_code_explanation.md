### 附录E：关键代码说明

#### E.1 数据抽样代码

```python
# 分层抽样：每月每类别抽取5000条
samples = []
for month in ['202501', '202502', ..., '202512']:
    for user_type in ['member', 'casual']:
        month_type_data = df[
            (df['month'] == month) & 
            (df['member_casual'] == user_type)
        ]
        sample = month_type_data.sample(
            n=5000, 
            random_state=42
        )
        samples.append(sample)

df_sampled = pd.concat(samples, ignore_index=True)
```

**设计考虑**：
- 固定随机种子42，保证可复现
- 每月每类别相同样本量，避免时间和类别偏差
- 总计120,000条样本（12月×2类×5000）

#### E.2 数据清洗代码

```python
# 1. 有效时间
df = df[df['ended_at'] > df['started_at']]

# 2. 计算时长和距离
df['duration_min'] = (
    df['ended_at'] - df['started_at']
).dt.total_seconds() / 60

df['distance_km'] = np.sqrt(
    ((df['end_lat'] - df['start_lat']) * 111)**2 + 
    ((df['end_lng'] - df['start_lng']) * 85)**2
)

# 3. 过滤异常值
df = df[
    (df['duration_min'] >= 1) & 
    (df['duration_min'] <= 180) &
    (df['distance_km'] >= 0.1) &
    (df['speed_kmh'] >= 2) &
    (df['speed_kmh'] <= 30)
]
```

**清洗规则依据**：
- 时长1-180分钟：短于1分钟可能是误触，长于3小时不合理
- 距离>0.1km：过短距离可能是定位误差
- 速度2-30km/h：低于2km/h不现实，高于30km/h超过骑行速度

#### E.3 目标编码代码

```python
from sklearn.model_selection import KFold

def target_encode_out_of_fold(df, col, target, n_folds=5):
    """防泄漏的目标编码"""
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(df):
        # 只用训练折计算编码
        train_stats = df.iloc[train_idx].groupby(col)[target].mean()
        # 应用到验证折
        encoded[val_idx] = df.iloc[val_idx][col].map(
            train_stats
        ).fillna(0.5)
    
    return encoded

# 对训练集使用out-of-fold
df_train['start_station_tm'] = target_encode_out_of_fold(
    df_train, 'start_station_id', 'is_member'
)

# 对测试集使用训练集统计
train_stats = df_train.groupby('start_station_id')['is_member'].mean()
df_test['start_station_tm'] = df_test['start_station_id'].map(
    train_stats
).fillna(0.5)
```

**防泄漏机制**：
- 训练集：5折交叉，用其他4折计算编码
- 测试集：只使用训练集的统计量
- 未见过的值：填充0.5（先验概率）
