#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from textwrap import dedent

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


RANDOM_STATE = 42
TARGET = "member_casual"
CLASS_ORDER = ["member", "casual"]
COORD_COLUMNS = ["start_lat", "start_lng", "end_lat", "end_lng"]

NUMERIC_FEATURES = [
    "duration_min",
    "distance_km",
    "speed_kmh",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
    "lat_delta",
    "lng_delta",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_commute_peak",
    "is_round_trip",
    "start_station_missing",
    "end_station_missing",
]

CATEGORICAL_FEATURES = [
    "rideable_type",
    "season",
    "time_period",
    "source_month",
    "hour_cat",
    "weekday_cat",
    "duration_bin",
    "distance_bin",
    "speed_bin",
    "start_grid",
    "end_grid",
    "route_grid",
    "start_station_id",
    "end_station_id",
]

TARGET_MEAN_FEATURES = [
    "start_station_id",
    "end_station_id",
    "route_station_id",
    "start_grid",
    "end_grid",
    "route_grid",
    "weekday_hour",
    "time_period",
    "rideable_type",
]

TARGET_MEAN_ENCODED_FEATURES = [f"{col}_target_mean" for col in TARGET_MEAN_FEATURES]
TARGET_MEAN_SOURCE_ONLY_FEATURES = ["route_station_id", "weekday_hour"]
MODEL_FEATURES = list(
    dict.fromkeys(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TARGET_MEAN_SOURCE_ONLY_FEATURES)
)

FEATURE_NAME_CN = {
    "duration_min": "骑行时长",
    "distance_km": "起终点直线距离",
    "speed_kmh": "估算骑行速度",
    "start_lat": "起点纬度",
    "start_lng": "起点经度",
    "end_lat": "终点纬度",
    "end_lng": "终点经度",
    "lat_delta": "南北方向位移",
    "lng_delta": "东西方向位移",
    "hour_sin": "小时周期特征",
    "hour_cos": "小时周期特征",
    "weekday_sin": "星期周期特征",
    "weekday_cos": "星期周期特征",
    "month_sin": "月份周期特征",
    "month_cos": "月份周期特征",
    "is_weekend": "是否周末",
    "is_commute_peak": "是否通勤高峰",
    "is_round_trip": "是否近似环线骑行",
    "start_station_missing": "起点站点是否缺失",
    "end_station_missing": "终点站点是否缺失",
    "rideable_type": "自行车类型",
    "season": "季节",
    "time_period": "一天内时间段",
    "source_month": "月份",
    "hour_cat": "开始小时",
    "weekday_cat": "星期类别",
    "duration_bin": "骑行时长分箱",
    "distance_bin": "距离分箱",
    "speed_bin": "速度分箱",
    "start_grid": "起点空间网格",
    "end_grid": "终点空间网格",
    "route_grid": "起终点网格组合",
    "start_station_id": "起点站点编号",
    "end_station_id": "终点站点编号",
    "route_station_id": "起终点站点编号组合",
    "weekday_hour": "星期与小时组合",
}


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "output": output_dir,
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
        "models": output_dir / "models",
        "processed": output_dir / "processed",
        "reports": output_dir.parent / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return dirs


def month_from_path(path: Path) -> str:
    match = re.search(r"(20\d{4})", str(path))
    if not match:
        return path.stem[:6]
    raw = match.group(1)
    return f"{raw[:4]}-{raw[4:]}"


def csv_files(data_dir: Path) -> list[Path]:
    files = sorted(data_dir.glob("*/*.csv"))
    if not files:
        raise FileNotFoundError(f"No monthly CSV files found under {data_dir}")
    return files


def profile_and_sample(
    files: list[Path],
    sample_per_month_class: int,
    random_state: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    raw_month_rows: list[dict[str, object]] = []
    raw_class_rows: list[dict[str, object]] = []
    raw_rideable_rows: list[dict[str, object]] = []
    raw_missing_rows: list[dict[str, object]] = []
    sample_frames: list[pd.DataFrame] = []

    for file_idx, path in enumerate(files):
        month = month_from_path(path)
        print(f"[1/6] Reading {month}: {path}")
        df = pd.read_csv(path, low_memory=False)
        df["source_month"] = month

        raw_month_rows.append({"source_month": month, "rows": int(len(df))})

        class_counts = df[TARGET].value_counts(dropna=False)
        for label, count in class_counts.items():
            raw_class_rows.append(
                {
                    "source_month": month,
                    "member_casual": str(label),
                    "count": int(count),
                }
            )

        rideable_counts = df["rideable_type"].value_counts(dropna=False)
        for label, count in rideable_counts.items():
            raw_rideable_rows.append(
                {
                    "source_month": month,
                    "rideable_type": str(label),
                    "count": int(count),
                }
            )

        for col in [
            "start_station_id",
            "end_station_id",
            "start_station_name",
            "end_station_name",
            *COORD_COLUMNS,
        ]:
            missing = df[col].isna() | (df[col].astype(str).str.strip() == "")
            raw_missing_rows.append(
                {
                    "source_month": month,
                    "column": col,
                    "missing_count": int(missing.sum()),
                    "missing_rate": float(missing.mean()),
                }
            )

        target_values = df[TARGET].astype(str).str.strip()
        for class_idx, label in enumerate(CLASS_ORDER):
            subset = df.loc[target_values == label]
            take = min(sample_per_month_class, len(subset))
            if take == 0:
                continue
            sampled = subset.sample(
                n=take,
                random_state=random_state + file_idx * 17 + class_idx,
                replace=False,
            )
            sample_frames.append(sampled)

    sampled = pd.concat(sample_frames, ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    tables = {
        "raw_month_rows": pd.DataFrame(raw_month_rows),
        "raw_month_class_counts": pd.DataFrame(raw_class_rows),
        "raw_month_rideable_counts": pd.DataFrame(raw_rideable_rows),
        "raw_missing_rates": pd.DataFrame(raw_missing_rows),
    }
    return sampled, tables


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(lat1.to_numpy(dtype=float))
    lon1_rad = np.radians(lon1.to_numpy(dtype=float))
    lat2_rad = np.radians(lat2.to_numpy(dtype=float))
    lon2_rad = np.radians(lon2.to_numpy(dtype=float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    return radius_km * 2.0 * np.arcsin(np.sqrt(a))


class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        columns: list[str] | None = None,
        smoothing: float = 20.0,
        n_splits: int = 5,
        random_state: int = RANDOM_STATE,
        positive_label: str = "casual",
    ):
        self.columns = columns or []
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.random_state = random_state
        self.positive_label = positive_label

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TargetMeanEncoder":
        y_num = (
            pd.Series(y).astype(str).eq(self.positive_label).astype(float).reset_index(drop=True)
        )
        self.global_mean_ = float(y_num.mean())
        self.maps_ = self._fit_maps(X.reset_index(drop=True), y_num)
        return self

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        **fit_params: object,
    ) -> pd.DataFrame:
        if y is None:
            raise ValueError("TargetMeanEncoder requires y during fit_transform.")

        X_reset = X.reset_index(drop=True)
        y_reset = pd.Series(y).astype(str).reset_index(drop=True)
        y_num = y_reset.eq(self.positive_label).astype(float)
        self.global_mean_ = float(y_num.mean())

        encoded = pd.DataFrame(index=X_reset.index)
        for col in self.columns:
            encoded[f"{col}_target_mean"] = self.global_mean_

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        for train_idx, valid_idx in skf.split(X_reset, y_reset):
            fold_maps = self._fit_maps(X_reset.iloc[train_idx], y_num.iloc[train_idx])
            for col in self.columns:
                encoded.loc[valid_idx, f"{col}_target_mean"] = self._map_column(
                    X_reset.iloc[valid_idx][col],
                    fold_maps[col],
                )

        self.maps_ = self._fit_maps(X_reset, y_num)
        return pd.concat([X_reset, encoded], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.reset_index(drop=True).copy()
        for col in self.columns:
            out[f"{col}_target_mean"] = self._map_column(out[col], self.maps_[col])
        return out

    def _fit_maps(self, X: pd.DataFrame, y_num: pd.Series) -> dict[str, dict[str, float]]:
        maps: dict[str, dict[str, float]] = {}
        for col in self.columns:
            tmp = pd.DataFrame(
                {
                    "category": X[col].astype("string").fillna("missing").reset_index(drop=True),
                    "target": y_num.reset_index(drop=True),
                }
            )
            grouped = tmp.groupby("category", observed=True)["target"].agg(["sum", "count"])
            encoded = (grouped["sum"] + self.smoothing * self.global_mean_) / (
                grouped["count"] + self.smoothing
            )
            maps[col] = encoded.to_dict()
        return maps

    def _map_column(self, values: pd.Series, mapping: dict[str, float]) -> pd.Series:
        return (
            values.astype("string")
            .fillna("missing")
            .map(mapping)
            .fillna(self.global_mean_)
            .astype(float)
        )


def add_cleaning_stage(rows: list[dict[str, object]], stage: str, before: int, after: int) -> None:
    rows.append(
        {
            "stage": stage,
            "rows_before": int(before),
            "rows_after": int(after),
            "removed": int(before - after),
            "removed_rate": float((before - after) / before) if before else 0.0,
        }
    )


def clean_and_engineer(sampled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = sampled.copy()
    cleaning_rows: list[dict[str, object]] = []

    before = len(df)
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df[df[TARGET].isin(CLASS_ORDER)].copy()
    add_cleaning_stage(cleaning_rows, "valid target class", before, len(df))

    before = len(df)
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df["ended_at"] = pd.to_datetime(df["ended_at"], errors="coerce")
    df = df.dropna(subset=["started_at", "ended_at"]).copy()
    add_cleaning_stage(cleaning_rows, "valid timestamps", before, len(df))

    before = len(df)
    for col in COORD_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=COORD_COLUMNS).copy()
    add_cleaning_stage(cleaning_rows, "valid coordinates", before, len(df))

    before = len(df)
    in_region = (
        df["start_lat"].between(38.65, 39.25)
        & df["end_lat"].between(38.65, 39.25)
        & df["start_lng"].between(-77.55, -76.70)
        & df["end_lng"].between(-77.55, -76.70)
    )
    df = df[in_region].copy()
    add_cleaning_stage(cleaning_rows, "coordinates inside DC-area bounds", before, len(df))

    before = len(df)
    df["duration_min"] = (df["ended_at"] - df["started_at"]).dt.total_seconds() / 60.0
    df = df[df["duration_min"].between(1, 180)].copy()
    add_cleaning_stage(cleaning_rows, "duration between 1 and 180 minutes", before, len(df))

    before = len(df)
    df["distance_km"] = haversine_km(
        df["start_lat"], df["start_lng"], df["end_lat"], df["end_lng"]
    )
    df["speed_kmh"] = np.where(
        df["duration_min"] > 0,
        df["distance_km"] / (df["duration_min"] / 60.0),
        np.nan,
    )
    plausible_motion = (df["distance_km"] <= 80) & (df["speed_kmh"].between(0, 45))
    df = df[plausible_motion].copy()
    add_cleaning_stage(cleaning_rows, "plausible distance and speed", before, len(df))

    before = len(df)
    if "ride_id" in df.columns:
        df = df.drop_duplicates(subset=["ride_id"]).copy()
    add_cleaning_stage(cleaning_rows, "drop duplicate ride_id", before, len(df))

    start_station_raw = df["start_station_id"].astype("string").str.strip()
    end_station_raw = df["end_station_id"].astype("string").str.strip()
    df["start_station_missing"] = (
        start_station_raw.isna() | (start_station_raw == "")
    ).astype(int)
    df["end_station_missing"] = (end_station_raw.isna() | (end_station_raw == "")).astype(
        int
    )
    same_station_id = (
        start_station_raw.notna()
        & end_station_raw.notna()
        & (start_station_raw != "")
        & (start_station_raw == end_station_raw)
    )
    df["is_round_trip"] = (same_station_id | (df["distance_km"] < 0.05)).astype(int)

    df["lat_delta"] = df["end_lat"] - df["start_lat"]
    df["lng_delta"] = df["end_lng"] - df["start_lng"]

    df["hour"] = df["started_at"].dt.hour + df["started_at"].dt.minute / 60.0
    df["weekday"] = df["started_at"].dt.weekday
    df["month"] = df["started_at"].dt.month
    df["hour_cat"] = df["started_at"].dt.hour.astype(int).astype(str).str.zfill(2)
    df["weekday_cat"] = df["weekday"].map(
        {
            0: "Mon",
            1: "Tue",
            2: "Wed",
            3: "Thu",
            4: "Fri",
            5: "Sat",
            6: "Sun",
        }
    )
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_commute_peak"] = (
        (df["weekday"] < 5)
        & (
            df["hour"].between(7, 9.5, inclusive="left")
            | df["hour"].between(16, 19, inclusive="left")
        )
    ).astype(int)

    season_map = {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
    df["season"] = df["month"].map(season_map)
    df["time_period"] = pd.cut(
        df["hour"],
        bins=[0, 5, 10, 16, 20, 24],
        labels=["late_night", "morning", "midday", "evening_peak", "night"],
        include_lowest=True,
        right=False,
    ).astype("string")
    df["duration_bin"] = pd.cut(
        df["duration_min"],
        bins=[1, 5, 10, 20, 40, 90, 180],
        labels=["01_5min", "05_10min", "10_20min", "20_40min", "40_90min", "90_180min"],
        include_lowest=True,
    ).astype("string")
    df["distance_bin"] = pd.cut(
        df["distance_km"],
        bins=[-0.001, 0.25, 0.75, 1.5, 3, 6, 80],
        labels=["zero_250m", "250_750m", "750m_1p5km", "1p5_3km", "3_6km", "6km_plus"],
        include_lowest=True,
    ).astype("string")
    df["speed_bin"] = pd.cut(
        df["speed_kmh"],
        bins=[-0.001, 5, 10, 15, 20, 30, 45],
        labels=["0_5", "5_10", "10_15", "15_20", "20_30", "30_45"],
        include_lowest=True,
    ).astype("string")
    df["start_grid"] = (
        df["start_lat"].round(2).astype(str) + "," + df["start_lng"].round(2).astype(str)
    )
    df["end_grid"] = (
        df["end_lat"].round(2).astype(str) + "," + df["end_lng"].round(2).astype(str)
    )
    df["route_grid"] = df["start_grid"] + "->" + df["end_grid"]

    for col in CATEGORICAL_FEATURES:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
            .fillna("missing")
        )

    df["weekday_hour"] = (
        df["weekday"].astype(int).astype(str) + "_" + df["hour_cat"].astype(str)
    )
    df["route_station_id"] = (
        df["start_station_id"].astype(str) + "->" + df["end_station_id"].astype(str)
    )
    for col in TARGET_MEAN_SOURCE_ONLY_FEATURES:
        df[col] = (
            df[col]
            .astype("string")
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "<NA>": pd.NA})
            .fillna("missing")
        )

    df = df.reset_index(drop=True)
    return df, pd.DataFrame(cleaning_rows)


def build_model() -> Pipeline:
    numeric_model_features = NUMERIC_FEATURES + TARGET_MEAN_ENCODED_FEATURES
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=30,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_model_features),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        sparse_threshold=0.35,
    )
    return Pipeline(
        steps=[
            (
                "target_mean",
                TargetMeanEncoder(
                    columns=TARGET_MEAN_FEATURES,
                    smoothing=20.0,
                    n_splits=5,
                    random_state=RANDOM_STATE,
                ),
            ),
            ("preprocess", preprocessor),
            (
                "model",
                LinearSVC(
                    class_weight="balanced",
                    dual="auto",
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def train_and_evaluate(
    df: pd.DataFrame,
    dirs: dict[str, Path],
) -> tuple[Pipeline, dict[str, object], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[MODEL_FEATURES].copy()
    y = df[TARGET].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    base_model = build_model()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base_model,
        param_grid={
            "model__C": [0.02, 0.03, 0.05, 0.1, 0.25, 0.5],
            "model__class_weight": [None, "balanced"],
        },
        scoring={"accuracy": "accuracy", "f1_weighted": "f1_weighted"},
        refit="f1_weighted",
        cv=cv,
        n_jobs=1,
        return_train_score=True,
    )

    print("[3/6] Training optimized Linear SVM with 3-fold CV on the 80% training split")
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_ORDER)

    metrics = {
        "random_state": RANDOM_STATE,
        "model": "TargetMeanEncoder + LinearSVC",
        "feature_count_raw": len(MODEL_FEATURES),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target_mean_features": TARGET_MEAN_FEATURES,
        "target_mean_encoded_features": TARGET_MEAN_ENCODED_FEATURES,
        "class_order": CLASS_ORDER,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_size": 0.2,
        "best_params": grid.best_params_,
        "best_cv_f1_weighted": float(grid.best_score_),
        "train_seconds": float(train_seconds),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    cv_results = pd.DataFrame(grid.cv_results_)
    cv_results.to_csv(dirs["tables"] / "svm_cv_results.csv", index=False)

    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "class"})
    report_df.to_csv(dirs["tables"] / "classification_report.csv", index=False)
    pd.DataFrame(
        cm,
        index=[f"actual_{x}" for x in CLASS_ORDER],
        columns=[f"pred_{x}" for x in CLASS_ORDER],
    ).to_csv(dirs["tables"] / "confusion_matrix.csv")

    joblib.dump(model, dirs["models"] / "svm_member_casual_pipeline.joblib")
    with (dirs["output"] / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return model, metrics, X_test, pd.Series(y_pred, index=y_test.index), y_train, y_test


def class_color(label: str) -> str:
    return {"member": "#2f6fbb", "casual": "#d65f5f"}.get(label, "#555555")


def save_stacked_month_bar(table: pd.DataFrame, title: str, path: Path) -> None:
    pivot = (
        table.pivot_table(
            index="source_month",
            columns="member_casual",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=CLASS_ORDER, fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for label in CLASS_ORDER:
        vals = pivot[label].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=label, color=class_color(label), width=0.72)
        bottom += vals
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Trips")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_sample_distribution(df: pd.DataFrame, path: Path) -> None:
    table = (
        df.groupby(["source_month", TARGET])
        .size()
        .reset_index(name="count")
        .sort_values(["source_month", TARGET])
    )
    save_stacked_month_bar(table, "Balanced Stratified Sample by Month and User Type", path)


def save_duration_boxplot(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    clipped = df.copy()
    upper = clipped["duration_min"].quantile(0.98)
    clipped["duration_min"] = clipped["duration_min"].clip(upper=upper)
    data = [clipped.loc[clipped[TARGET] == label, "duration_min"] for label in CLASS_ORDER]
    box = ax.boxplot(data, tick_labels=CLASS_ORDER, showfliers=False, patch_artist=True)
    for patch, label in zip(box["boxes"], CLASS_ORDER):
        patch.set_facecolor(class_color(label))
    ax.set_title("Trip Duration Distribution by User Type")
    ax.set_xlabel("User Type")
    ax.set_ylabel("Duration (minutes, clipped at 98th percentile)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_rideable_distribution(df: pd.DataFrame, path: Path) -> None:
    ctab = pd.crosstab(df["rideable_type"], df[TARGET], normalize="columns")
    ctab = ctab.reindex(columns=CLASS_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    x = np.arange(len(ctab))
    width = 0.35
    for idx, label in enumerate(CLASS_ORDER):
        ax.bar(
            x + (idx - 0.5) * width,
            ctab[label] * 100,
            width=width,
            label=label,
            color=class_color(label),
        )
    ax.set_title("Rideable Type Share by User Type")
    ax.set_xlabel("Rideable Type")
    ax.set_ylabel("Share within user type (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(ctab.index, rotation=25, ha="right")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_hour_weekday_heatmap(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150, sharey=True)
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for ax, label in zip(axes, CLASS_ORDER):
        subset = df[df[TARGET] == label]
        heat = pd.crosstab(subset["weekday"], subset["started_at"].dt.hour)
        heat = heat.reindex(index=range(7), columns=range(24), fill_value=0)
        heat_pct = heat / max(1, heat.to_numpy().sum()) * 100
        im = ax.imshow(heat_pct, aspect="auto", cmap="YlGnBu")
        ax.set_title(f"{label}: Hour x Weekday Share")
        ax.set_xlabel("Hour")
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(7))
        ax.set_yticklabels(weekday_labels)
    axes[0].set_ylabel("Weekday")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
    cbar.set_label("Share of trips (%)")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_spatial_scatter(df: pd.DataFrame, path: Path, random_state: int) -> None:
    plot_df = df.sample(n=min(16000, len(df)), random_state=random_state)
    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)
    for label in CLASS_ORDER:
        subset = plot_df[plot_df[TARGET] == label]
        ax.scatter(
            subset["start_lng"],
            subset["start_lat"],
            s=5,
            alpha=0.25,
            label=label,
            color=class_color(label),
            linewidths=0,
        )
    ax.set_title("Start Location Distribution in Stratified Sample")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(markerscale=3, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_confusion_matrix(cm: list[list[int]], path: Path) -> None:
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title("SVM Confusion Matrix")
    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_yticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels([f"Pred {x}" for x in CLASS_ORDER])
    ax.set_yticklabels([f"Actual {x}" for x in CLASS_ORDER])
    total = arr.sum()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            pct = arr[i, j] / total * 100 if total else 0
            ax.text(j, i, f"{arr[i, j]}\n{pct:.1f}%", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_model_comparison(cv_results: pd.DataFrame, path: Path) -> None:
    cols = ["param_model__C", "mean_test_accuracy", "mean_test_f1_weighted"]
    if "param_model__class_weight" in cv_results.columns:
        cols.append("param_model__class_weight")
    df = cv_results[cols].copy()
    labels = []
    for row in df.itertuples(index=False):
        class_weight = getattr(row, "param_model__class_weight", None)
        weight_label = "balanced" if class_weight == "balanced" else "none"
        labels.append(f"C={row.param_model__C}\nweight={weight_label}")
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    ax.bar(x - width / 2, df["mean_test_accuracy"], width=width, label="CV accuracy")
    ax.bar(x + width / 2, df["mean_test_f1_weighted"], width=width, label="CV weighted F1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("SVM Hyperparameter Comparison on Training Split")
    ax.set_xlabel("LinearSVC C")
    ax.set_ylabel("3-fold CV score")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_flowchart(labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=150)
    ax.set_axis_off()
    xs = np.linspace(0.08, 0.92, len(labels))
    y = 0.52
    box_w = min(0.125, 0.62 / len(labels))
    box_h = 0.28
    for i, (x, label) in enumerate(zip(xs, labels)):
        box = FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=1.2,
            edgecolor="#335c81",
            facecolor="#eef5fb",
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha="center", va="center", fontsize=9)
        if i < len(labels) - 1:
            arrow = FancyArrowPatch(
                (x + box_w / 2 + 0.01, y),
                (xs[i + 1] - box_w / 2 - 0.01, y),
                arrowstyle="->",
                mutation_scale=14,
                linewidth=1.2,
                color="#4a4a4a",
            )
            ax.add_patch(arrow)
    ax.text(0.5, 0.9, title, ha="center", va="center", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(importance: pd.DataFrame, path: Path) -> None:
    top = importance.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["#4c78a8" if x >= 0 else "#f58518" for x in top["importance_mean"]]
    ax.barh(top["feature"], top["importance_mean"], color=colors)
    ax.set_title("Permutation Importance on Test Subset")
    ax.set_xlabel("Mean accuracy decrease")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def generate_plots(
    raw_tables: dict[str, pd.DataFrame],
    clean_df: pd.DataFrame,
    metrics: dict[str, object],
    importance: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    print("[5/6] Creating figures")
    save_stacked_month_bar(
        raw_tables["raw_month_class_counts"],
        "Raw Monthly Trips by User Type",
        dirs["figures"] / "01_raw_monthly_user_type.png",
    )
    save_sample_distribution(clean_df, dirs["figures"] / "02_clean_sample_distribution.png")
    save_duration_boxplot(clean_df, dirs["figures"] / "03_duration_by_user_type.png")
    save_rideable_distribution(clean_df, dirs["figures"] / "04_rideable_type_share.png")
    save_hour_weekday_heatmap(clean_df, dirs["figures"] / "05_hour_weekday_heatmap.png")
    save_spatial_scatter(
        clean_df,
        dirs["figures"] / "06_start_location_scatter.png",
        random_state=RANDOM_STATE,
    )
    save_confusion_matrix(
        metrics["confusion_matrix"],
        dirs["figures"] / "07_confusion_matrix.png",
    )
    save_feature_importance(importance, dirs["figures"] / "08_permutation_importance.png")
    cv_results = pd.read_csv(dirs["tables"] / "svm_cv_results.csv")
    save_model_comparison(cv_results, dirs["figures"] / "09_svm_cv_comparison.png")
    draw_flowchart(
        [
            "Raw CSV",
            "Monthly\nstratified\nsampling",
            "Cleaning",
            "Feature\nengineering",
            "80/20 split",
            "SVM training",
            "Evaluation",
        ],
        "Experiment Workflow",
        dirs["figures"] / "10_experiment_workflow.png",
    )
    draw_flowchart(
        [
            "Feature\nvector x",
            "Target mean +\nScale + One-Hot",
            "Max-margin\noptimization",
            "Decision\nfunction",
            "member /\ncasual",
        ],
        "SVM Classification Workflow",
        dirs["figures"] / "11_svm_workflow.png",
    )


def build_eda_summary(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {}
    duration = df.groupby(TARGET)["duration_min"].agg(["mean", "median"]).round(3)
    summary["duration"] = duration.to_dict(orient="index")
    summary["weekend_share"] = (
        df.groupby(TARGET)["is_weekend"].mean().mul(100).round(2).to_dict()
    )
    summary["commute_peak_share"] = (
        df.groupby(TARGET)["is_commute_peak"].mean().mul(100).round(2).to_dict()
    )
    summary["round_trip_share"] = (
        df.groupby(TARGET)["is_round_trip"].mean().mul(100).round(2).to_dict()
    )
    summary["station_missing_share"] = (
        df.groupby(TARGET)[["start_station_missing", "end_station_missing"]]
        .mean()
        .mul(100)
        .round(2)
        .to_dict(orient="index")
    )
    rideable = pd.crosstab(df["rideable_type"], df[TARGET], normalize="columns").mul(100)
    summary["rideable_share"] = rideable.round(2).to_dict()

    peak_rows = []
    for label in CLASS_ORDER:
        hour_counts = df.loc[df[TARGET] == label, "started_at"].dt.hour.value_counts()
        peak_rows.append(
            {
                "member_casual": label,
                "peak_hour": int(hour_counts.idxmax()),
                "peak_hour_count": int(hour_counts.max()),
            }
        )
    summary["peak_hours"] = peak_rows
    return summary


def compute_permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    dirs: dict[str, Path],
    rows: int,
) -> pd.DataFrame:
    print("[4/6] Computing permutation importance on a held-out subset")
    tmp = X_test.copy()
    tmp[TARGET] = y_test.to_numpy()
    if len(tmp) > rows:
        per_class = max(1, rows // len(CLASS_ORDER))
        parts = []
        for label in CLASS_ORDER:
            subset = tmp[tmp[TARGET] == label]
            parts.append(
                subset.sample(
                    n=min(per_class, len(subset)),
                    random_state=RANDOM_STATE,
                    replace=False,
                )
            )
        tmp = pd.concat(parts, ignore_index=True)
        tmp = tmp.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    X_imp = tmp.drop(columns=[TARGET])
    y_imp = tmp[TARGET]
    result = permutation_importance(
        model,
        X_imp,
        y_imp,
        scoring="accuracy",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X_imp.columns,
            "feature_cn": [FEATURE_NAME_CN.get(x, x) for x in X_imp.columns],
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(dirs["tables"] / "permutation_importance.csv", index=False)
    return importance


def write_report_outline(
    raw_tables: dict[str, pd.DataFrame],
    clean_df: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    metrics: dict[str, object],
    importance: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    eda = build_eda_summary(clean_df)
    raw_total = int(raw_tables["raw_month_rows"]["rows"].sum())
    raw_months = len(raw_tables["raw_month_rows"])
    cleaned_rows = int(len(clean_df))
    train_rows = metrics["train_rows"]
    test_rows = metrics["test_rows"]
    best_c = metrics["best_params"].get("model__C")
    best_class_weight = metrics["best_params"].get("model__class_weight")
    class_weight_text = "None" if best_class_weight is None else str(best_class_weight)
    acc = metrics["accuracy"]
    f1w = metrics["f1_weighted"]
    precision = metrics["precision_weighted"]
    recall = metrics["recall_weighted"]

    member_duration = eda["duration"].get("member", {})
    casual_duration = eda["duration"].get("casual", {})
    member_weekend = eda["weekend_share"].get("member", 0.0)
    casual_weekend = eda["weekend_share"].get("casual", 0.0)
    member_commute = eda["commute_peak_share"].get("member", 0.0)
    casual_commute = eda["commute_peak_share"].get("casual", 0.0)
    member_round = eda["round_trip_share"].get("member", 0.0)
    casual_round = eda["round_trip_share"].get("casual", 0.0)
    peak_map = {row["member_casual"]: row["peak_hour"] for row in eda["peak_hours"]}

    top_features = importance.head(8).copy()
    top_feature_lines = "\n".join(
        [
            f"- {row.feature_cn}（`{row.feature}`）：置换后准确率平均下降 {row.importance_mean:.4f}"
            for row in top_features.itertuples()
        ]
    )

    cleaning_lines = "\n".join(
        [
            f"- {row.stage}: {int(row.rows_before)} -> {int(row.rows_after)}, 删除 {int(row.removed)} 条"
            for row in cleaning_summary.itertuples()
        ]
    )

    report = f"""# 基于 SVM 的 Capital Bikeshare 用户类型分类实验报告提纲

## 1. 实验问题与目标

本实验使用 Capital Bikeshare 公布的 2025 年 1 月至 12 月华盛顿都会区共享单车轨迹数据，目标是根据一次骑行的时间、空间、车辆类型和站点等信息，预测用户类型 `member_casual`，即会员用户 `member` 或散户用户 `casual`。该问题是一个二分类任务，适合使用支持向量机（SVM）构建最大间隔分类器。

实验目标可以写为：

1. 按实验要求完成变量筛选、缺失值处理、异常值过滤、归一化/标准化和类别变量编码。
2. 按月份和用户类别进行分层抽样，保证 12 个月都有样本，且正负样本尽量均衡。
3. 使用 SVM 完成会员/散户分类预测，并使用 80%:20% 的训练测试划分验证模型效果。
4. 分析时间、空间、车辆类型、站点缺失等因素对用户类型的影响，给出可解释的实验发现。

## 2. 数据介绍与抽样方案

原始数据共有 {raw_months} 个月、{raw_total:,} 条骑行记录。字段包括 `ride_id`、`rideable_type`、`started_at`、`ended_at`、起终点站点名称与编号、起终点经纬度以及目标变量 `member_casual`。

本实验没有直接使用全部原始数据训练模型，而是按月、按类别进行分层抽样：每个月分别从 `member` 和 `casual` 中抽取相同规模样本，随后再清洗。这样做有两个原因：第一，原始数据规模超过 600 万行，直接训练 SVM 成本较高；第二，原始会员和散户比例不完全均衡，分层抽样可以避免模型过度偏向多数类。最终清洗后样本量为 {cleaned_rows:,} 条，其中训练集 {train_rows:,} 条，测试集 {test_rows:,} 条。

建议在报告中配图：

- `outputs/figures/01_raw_monthly_user_type.png`：原始各月会员/散户数量分布。
- `outputs/figures/02_clean_sample_distribution.png`：清洗后分层样本在 12 个月中的分布。
- `outputs/figures/06_start_location_scatter.png`：起点空间分布。

## 3. 数据预处理与特征工程

清洗过程：

{cleaning_lines}

主要特征工程如下：

- 时间特征：从 `started_at` 中提取小时、星期、月份，构造周期特征 `sin/cos`，并标记周末和工作日通勤高峰。
- 空间特征：使用起终点经纬度计算直线距离、南北/东西位移、估算速度，同时构造起点、终点和路线空间网格。
- 行程特征：使用 `ended_at - started_at` 计算骑行时长，标记近似环线骑行，并加入时长、距离和速度分箱。
- 车辆与站点特征：保留 `rideable_type`、起终点站点编号、站点缺失标记、月份、季节、小时类别、星期类别和时间段。
- 目标编码特征：对起点站、终点站、站点路线、起终点网格路线、星期小时组合、时间段和车辆类型构造平滑 target mean 编码；训练集内部使用 out-of-fold 方式生成编码，测试集只使用训练集统计映射，避免数据泄漏。
- 数值变量使用中位数填补和标准化；类别变量使用缺失值填补与 One-Hot 编码，并保留低频类别阈值对模型效果的影响。

## 4. SVM 方法说明

本实验使用线性 SVM（`LinearSVC`）。其核心思想是寻找一个能最大化两类样本间隔的分类超平面。软间隔 SVM 的优化目标可写为：

```text
min  1/2 ||w||^2 + C * sum(xi_i)
s.t. y_i * (w^T phi(x_i) + b) >= 1 - xi_i, xi_i >= 0
```

其中 `C` 是正则化参数，控制间隔宽度与训练误差之间的权衡。本实验在训练集上使用 3 折交叉验证比较 `C = 0.02, 0.03, 0.05, 0.1, 0.25, 0.5`，并同时比较 `class_weight=None` 与 `class_weight="balanced"`，最终最优参数为 `C = {best_c}`、`class_weight = {class_weight_text}`。实验流程图可放入 `outputs/figures/10_experiment_workflow.png`，SVM 算法流程图可放入 `outputs/figures/11_svm_workflow.png`。

## 5. 验证过程与实验结果

训练和预测比例严格采用 80%:20%，并使用分层划分保证训练集和测试集中会员/散户比例一致。测试集结果如下：

- Accuracy: {acc:.4f}
- Weighted Precision: {precision:.4f}
- Weighted Recall: {recall:.4f}
- Weighted F1: {f1w:.4f}
- 3 折交叉验证最优 Weighted F1: {metrics["best_cv_f1_weighted"]:.4f}

建议在报告中配图：

- `outputs/figures/07_confusion_matrix.png`：测试集混淆矩阵。
- `outputs/figures/09_svm_cv_comparison.png`：不同 `C` 参数下的交叉验证结果。
- `outputs/tables/classification_report.csv`：每一类的 precision、recall、F1。

## 6. 因素影响与有趣发现

### 发现 1：骑行时长是区分会员和散户的重要线索

在清洗后的样本中，会员平均骑行时长约为 {member_duration.get("mean", 0):.2f} 分钟，中位数约为 {member_duration.get("median", 0):.2f} 分钟；散户平均骑行时长约为 {casual_duration.get("mean", 0):.2f} 分钟，中位数约为 {casual_duration.get("median", 0):.2f} 分钟。散户通常更容易出现较长骑行，符合旅游、休闲、临时出行的使用场景。该现象可以结合 `outputs/figures/03_duration_by_user_type.png` 说明。

### 发现 2：会员更接近通勤型使用，散户更接近休闲型使用

会员周末骑行占比约为 {member_weekend:.2f}%，散户周末骑行占比约为 {casual_weekend:.2f}%。会员工作日通勤高峰占比约为 {member_commute:.2f}%，散户约为 {casual_commute:.2f}%。会员峰值小时为 {peak_map.get("member", 0)} 点，散户峰值小时为 {peak_map.get("casual", 0)} 点。该结果说明时间结构对用户类型有明显解释力，可以结合 `outputs/figures/05_hour_weekday_heatmap.png` 分析。

### 发现 3：环线或近距离返回原点的骑行更偏休闲属性

会员近似环线骑行占比约为 {member_round:.2f}%，散户约为 {casual_round:.2f}%。如果散户占比更高，可以解释为散户更常出现观光、短途体验、往返同一地点等骑行；如果会员占比更高，则可解释为部分会员短距离接驳或同站还车行为更明显。

### 发现 4：车辆类型、站点信息缺失和空间位置共同影响分类

置换重要性排名靠前的原始特征如下：

{top_feature_lines}

其中，时间和行程长度类特征通常反映出“通勤 vs 休闲”的差异；站点编号和空间位置则反映城市功能区差异，例如办公区、居住区、景区和交通枢纽的用车人群不同。站点缺失特征也值得关注，因为电动自行车或非固定桩场景下站点字段更容易缺失，这会间接携带车辆类型和使用场景信息。

## 7. 可写入正式报告的优点说明

- 抽样方法严格保留 12 个月时间代表性，同时平衡会员/散户类别，降低类别不均衡对模型的影响。
- 预处理覆盖缺失值、异常时间、异常坐标、异常速度和重复记录，数据质量控制较完整。
- 特征不仅包含原始字段，还加入时间周期、通勤高峰、空间距离、环线骑行、路线组合和防泄漏 target mean 编码等变量。
- 验证过程采用固定随机种子、分层 80%:20% 划分和训练集 3 折交叉验证，结果可复现。
- 除准确率外，还记录 precision、recall、F1、混淆矩阵和置换重要性，便于从多个角度评价模型。

## 8. 报告结构建议

1. 实验问题与研究意义：说明共享单车用户类型预测的实际意义。
2. 数据来源与字段说明：介绍 2025 年 Capital Bikeshare 数据和目标变量。
3. 数据抽样与预处理：说明按月、按类别分层抽样，以及清洗规则。
4. 特征工程：说明时间、空间、行程、车辆、站点特征。
5. SVM 算法原理：写出最大间隔思想、软间隔公式和参数 `C` 的含义。
6. 实验流程与验证方案：加入流程图，说明 80%:20% 划分和交叉验证。
7. 实验结果：展示准确率、F1、分类报告、混淆矩阵。
8. 结果分析与发现：围绕时长、通勤高峰、周末、空间分布、车辆类型展开。
9. 总结与改进：说明 SVM 在该任务上的有效性，也可提出未来加入天气、节假日、真实道路距离或更多核函数对比。

## 9. 已生成文件索引

- 模型文件：`outputs/models/svm_member_casual_pipeline.joblib`
- 核心指标：`outputs/metrics.json`
- 清洗后样本：`outputs/processed/cleaned_stratified_sample.csv`
- 交叉验证记录：`outputs/tables/svm_cv_results.csv`
- 优化实验对比：`outputs/tables/optimization_experiments.csv`、`outputs/tables/optimization_extra_experiments.csv`
- 特征重要性：`outputs/tables/permutation_importance.csv`
- 图表目录：`outputs/figures/`
"""

    report_path = dirs["reports"] / "report_outline.md"
    report_path.write_text(report, encoding="utf-8")


def save_tables(
    raw_tables: dict[str, pd.DataFrame],
    sampled: pd.DataFrame,
    clean_df: pd.DataFrame,
    cleaning_summary: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    for name, table in raw_tables.items():
        table.to_csv(dirs["tables"] / f"{name}.csv", index=False)

    sample_counts = (
        sampled.groupby(["source_month", TARGET])
        .size()
        .reset_index(name="count")
        .sort_values(["source_month", TARGET])
    )
    sample_counts.to_csv(dirs["tables"] / "sample_before_cleaning_counts.csv", index=False)

    clean_counts = (
        clean_df.groupby(["source_month", TARGET])
        .size()
        .reset_index(name="count")
        .sort_values(["source_month", TARGET])
    )
    clean_counts.to_csv(dirs["tables"] / "clean_sample_counts.csv", index=False)
    cleaning_summary.to_csv(dirs["tables"] / "cleaning_summary.csv", index=False)
    clean_df.to_csv(dirs["processed"] / "cleaned_stratified_sample.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Capital Bikeshare member/casual SVM experiment."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--sample-per-month-class",
        type=int,
        default=2500,
        help="Number of member and casual rows sampled from each month before cleaning.",
    )
    parser.add_argument(
        "--importance-rows",
        type=int,
        default=6000,
        help="Held-out rows used for permutation importance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    dirs = ensure_dirs(output_dir)

    files = csv_files(data_dir)
    print(f"Found {len(files)} monthly CSV files.")

    sampled, raw_tables = profile_and_sample(
        files=files,
        sample_per_month_class=args.sample_per_month_class,
        random_state=RANDOM_STATE,
    )
    print(f"[2/6] Sampled {len(sampled):,} rows before cleaning")
    clean_df, cleaning_summary = clean_and_engineer(sampled)
    print(f"[2/6] Cleaned sample has {len(clean_df):,} rows")
    save_tables(raw_tables, sampled, clean_df, cleaning_summary, dirs)

    model, metrics, X_test, y_pred, y_train, y_test = train_and_evaluate(clean_df, dirs)
    importance = compute_permutation_importance(
        model,
        X_test,
        y_test,
        dirs,
        rows=args.importance_rows,
    )
    generate_plots(raw_tables, clean_df, metrics, importance, dirs)
    print("[6/6] Writing report outline")
    write_report_outline(raw_tables, clean_df, cleaning_summary, metrics, importance, dirs)

    print(
        dedent(
            f"""
            Done.
            Accuracy: {metrics['accuracy']:.4f}
            Weighted F1: {metrics['f1_weighted']:.4f}
            Best params: {metrics['best_params']}
            Report outline: {dirs['reports'] / 'report_outline.md'}
            Metrics: {dirs['output'] / 'metrics.json'}
            Figures: {dirs['figures']}
            """
        ).strip()
    )


if __name__ == "__main__":
    main()
