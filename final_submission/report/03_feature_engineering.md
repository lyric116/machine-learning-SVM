### 3.2 数据预处理流程

#### 3.2.1 特征工程

从原始13个字段出发，本实验构造了以下5类共37个特征：

**1. 基础特征（5个）**
- `duration_min`：骑行时长（分钟）= (ended_at - started_at) / 60
- `distance_km`：直线距离（公里）= $\sqrt{(\Delta lat \times 111)^2 + (\Delta lng \times 85)^2}$
- `speed_kmh`：估算速度（公里/小时）= distance_km / (duration_min / 60)
- `weekday`：星期几（0=周一，6=周日）
- `hour`：骑行开始小时（0-23）

**2. 时间特征（12个）**
- `month`：月份（1-12）
- `is_weekend`：是否周末（0/1）
- `is_commute_peak`：是否通勤高峰（7-9点或17-19点）
- `hour_sin`, `hour_cos`：小时的周期编码
- `weekday_sin`, `weekday_cos`：星期的周期编码
- `time_of_day`：时间段（morning/noon/afternoon/evening/night）
- `season`：季节（spring/summer/fall/winter）
- `hour_category`：小时类别（dawn/morning/noon/afternoon/evening/night）

**3. 空间特征（6个）**
- `start_grid`, `end_grid`：起终点空间网格编码
- `route_grid`：路线网格组合
- `heading_ns`：南北方向位移（度）
- `heading_ew`：东西方向位移（度）
- `is_circular`：是否近似环线骑行（起终点距离<0.1km）

**4. 站点特征（5个）**
- `start_station_missing`：起点站点信息是否缺失
- `end_station_missing`：终点站点信息是否缺失
- `route_station_id`：起终点站点ID组合
- `start_station_id`, `end_station_id`：站点ID

**5. 目标编码特征（9个）**

这是本实验的**关键创新**：计算各站点/路线的会员使用比例
- `start_station_target_mean`：起点站的会员比例
- `end_station_target_mean`：终点站的会员比例
- `route_target_mean`：路线的会员比例
- `start_grid_target_mean`：起点网格的会员比例
- `end_grid_target_mean`：终点网格的会员比例
- `route_grid_target_mean`：路线网格的会员比例
- `weekday_hour_target_mean`：星期×小时的会员比例
- `time_category_target_mean`：时间段的会员比例
- `rideable_target_mean`：车辆类型的会员比例

**防止数据泄漏**：目标编码使用out-of-fold策略
- 训练集：5折交叉验证，用其他4折计算编码
- 测试集：只使用训练集的统计量
