#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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

BASE_NUMERIC_FEATURES = [
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

BASE_CATEGORICAL_FEATURES = [
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


def haversine_np(lat1: pd.Series, lon1: pd.Series, lat2: object, lon2: object) -> np.ndarray:
    radius_km = 6371.0088
    lat1_rad = np.radians(pd.to_numeric(lat1, errors="coerce").to_numpy(dtype=float))
    lon1_rad = np.radians(pd.to_numeric(lon1, errors="coerce").to_numpy(dtype=float))
    lat2_rad = np.radians(np.asarray(lat2, dtype=float))
    lon2_rad = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        dlon / 2.0
    ) ** 2
    return radius_km * 2.0 * np.arcsin(np.sqrt(a))


def add_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["daytype"] = np.where(out["is_weekend"].astype(int) == 1, "weekend", "workday")
    out["weekday_hour"] = out["weekday"].astype(int).astype(str) + "_" + out["hour_cat"].astype(str)
    out["daytype_time_period"] = out["daytype"].astype(str) + "_" + out["time_period"].astype(str)
    out["month_time_period"] = out["month"].astype(int).astype(str) + "_" + out["time_period"].astype(str)
    out["season_time_period"] = out["season"].astype(str) + "_" + out["time_period"].astype(str)
    out["rideable_time_period"] = out["rideable_type"].astype(str) + "_" + out["time_period"].astype(str)
    out["rideable_daytype"] = out["rideable_type"].astype(str) + "_" + out["daytype"].astype(str)

    out["abs_lat_delta"] = out["lat_delta"].abs()
    out["abs_lng_delta"] = out["lng_delta"].abs()
    trip_angle = np.arctan2(out["lat_delta"], out["lng_delta"])
    out["trip_angle_sin"] = np.sin(trip_angle)
    out["trip_angle_cos"] = np.cos(trip_angle)
    out["trip_direction_bin"] = pd.cut(
        trip_angle,
        bins=np.linspace(-np.pi, np.pi, 9),
        labels=[f"dir_{i}" for i in range(8)],
        include_lowest=True,
    ).astype("string")
    out["move_type"] = np.select(
        [
            out["abs_lat_delta"] < 0.002,
            out["abs_lng_delta"] < 0.002,
            out["abs_lat_delta"] > out["abs_lng_delta"] * 1.5,
            out["abs_lng_delta"] > out["abs_lat_delta"] * 1.5,
        ],
        [
            "mostly_east_west",
            "mostly_north_south",
            "vertical_dominant",
            "horizontal_dominant",
        ],
        default="diagonal",
    )

    out["duration_log1p"] = np.log1p(out["duration_min"].clip(lower=0))
    out["distance_log1p"] = np.log1p(out["distance_km"].clip(lower=0))
    out["speed_log1p"] = np.log1p(out["speed_kmh"].clip(lower=0))
    out["duration_per_km"] = out["duration_min"] / (out["distance_km"] + 1e-3)
    out["distance_per_min"] = out["distance_km"] / (out["duration_min"] + 1e-3)

    out["same_grid"] = (out["start_grid"].astype(str) == out["end_grid"].astype(str)).astype(str)
    out["route_station_id"] = (
        out["start_station_id"].astype("string").fillna("missing").astype(str)
        + "->"
        + out["end_station_id"].astype("string").fillna("missing").astype(str)
    )
    out["rideable_duration_bin"] = out["rideable_type"].astype(str) + "_" + out["duration_bin"].astype(str)
    out["rideable_distance_bin"] = out["rideable_type"].astype(str) + "_" + out["distance_bin"].astype(str)
    out["weekend_duration_bin"] = out["daytype"].astype(str) + "_" + out["duration_bin"].astype(str)
    out["time_direction"] = out["time_period"].astype(str) + "_" + out["trip_direction_bin"].astype(str)
    out["same_grid_time"] = out["same_grid"].astype(str) + "_" + out["time_period"].astype(str)

    return out


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FrequencyEncoder":
        self.freq_maps_ = {}
        n = len(X)
        for col in self.columns:
            values = X[col].astype("string").fillna("missing")
            self.freq_maps_[col] = (values.value_counts() / max(1, n)).to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.columns:
            out[f"{col}_freq"] = (
                out[col].astype("string").fillna("missing").map(self.freq_maps_[col]).fillna(0.0)
            )
        return out


class DataCenterDistanceFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "DataCenterDistanceFeatures":
        lat_values = pd.concat([X["start_lat"], X["end_lat"]], ignore_index=True)
        lng_values = pd.concat([X["start_lng"], X["end_lng"]], ignore_index=True)
        self.center_lat_ = float(np.nanmedian(pd.to_numeric(lat_values, errors="coerce")))
        self.center_lng_ = float(np.nanmedian(pd.to_numeric(lng_values, errors="coerce")))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out["start_center_dist"] = haversine_np(
            out["start_lat"], out["start_lng"], self.center_lat_, self.center_lng_
        )
        out["end_center_dist"] = haversine_np(
            out["end_lat"], out["end_lng"], self.center_lat_, self.center_lng_
        )
        out["center_dist_diff"] = out["end_center_dist"] - out["start_center_dist"]
        out["center_dist_abs_diff"] = out["center_dist_diff"].abs()
        return out


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
        y_num = (pd.Series(y).astype(str) == self.positive_label).astype(float).reset_index(drop=True)
        self.global_mean_ = float(y_num.mean())
        self.maps_ = self._fit_maps(X.reset_index(drop=True), y_num)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **fit_params) -> pd.DataFrame:
        if y is None:
            return self.fit(X, y).transform(X)

        X_reset = X.reset_index(drop=True)
        y_num = (pd.Series(y).astype(str) == self.positive_label).astype(float).reset_index(drop=True)
        self.global_mean_ = float(y_num.mean())
        encoded = pd.DataFrame(index=X_reset.index)
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        strat_y = pd.Series(y).astype(str).reset_index(drop=True)
        for col in self.columns:
            encoded[f"{col}_target_mean"] = self.global_mean_
        for train_idx, valid_idx in skf.split(X_reset, strat_y):
            fold_maps = self._fit_maps(X_reset.iloc[train_idx], y_num.iloc[train_idx])
            for col in self.columns:
                encoded.loc[valid_idx, f"{col}_target_mean"] = self._map_column(
                    X_reset.iloc[valid_idx][col], fold_maps[col]
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
        return values.astype("string").fillna("missing").map(mapping).fillna(self.global_mean_).astype(float)


@dataclass
class Experiment:
    name: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    frequency: list[str] = field(default_factory=list)
    target_mean: list[str] = field(default_factory=list)
    use_center: bool = False
    min_frequency: int | None = 30
    param_grid: dict[str, list[object]] = field(
        default_factory=lambda: {"model__C": [0.1, 0.25, 0.5, 1.0, 2.0]}
    )
    notes: str = ""


def make_pipeline(exp: Experiment) -> Pipeline:
    numeric_features = BASE_NUMERIC_FEATURES + exp.numeric
    categorical_features = BASE_CATEGORICAL_FEATURES + exp.categorical

    for col in exp.frequency:
        numeric_features.append(f"{col}_freq")
    for col in exp.target_mean:
        numeric_features.append(f"{col}_target_mean")
    if exp.use_center:
        numeric_features.extend(
            ["start_center_dist", "end_center_dist", "center_dist_diff", "center_dist_abs_diff"]
        )

    steps: list[tuple[str, object]] = []
    if exp.frequency:
        steps.append(("freq", FrequencyEncoder(exp.frequency)))
    if exp.target_mean:
        steps.append(("target_mean", TargetMeanEncoder(exp.target_mean)))
    if exp.use_center:
        steps.append(("center", DataCenterDistanceFeatures()))

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
                    min_frequency=exp.min_frequency,
                    sparse_output=True,
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        sparse_threshold=0.35,
    )
    steps.extend(
        [
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
    return Pipeline(steps)


def metric_row(
    exp: Experiment,
    grid: GridSearchCV,
    y_test: pd.Series,
    y_pred: np.ndarray,
    train_seconds: float,
    test_seconds: float,
) -> dict[str, object]:
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "experiment_name": exp.name,
        "model": "LinearSVC",
        "best_params": json.dumps(grid.best_params_, ensure_ascii=False),
        "best_cv_weighted_f1": float(grid.best_score_),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(
            precision_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "member_f1": float(report["member"]["f1-score"]),
        "casual_precision": float(report["casual"]["precision"]),
        "casual_recall": float(report["casual"]["recall"]),
        "casual_f1": float(report["casual"]["f1-score"]),
        "train_time": train_seconds,
        "test_time": test_seconds,
        "notes": exp.notes,
    }


def run_experiment(exp: Experiment, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, object]:
    model = make_pipeline(exp)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        model,
        param_grid=exp.param_grid,
        scoring="f1_weighted",
        refit=True,
        cv=cv,
        n_jobs=1,
        return_train_score=True,
    )
    start = time.perf_counter()
    grid.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = grid.predict(X_test)
    test_seconds = time.perf_counter() - start
    return metric_row(exp, grid, y_test, y_pred, train_seconds, test_seconds)


def main() -> None:
    output_dir = Path("outputs")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(output_dir / "processed" / "cleaned_stratified_sample.csv")
    df = add_candidate_features(df)
    for col in BASE_CATEGORICAL_FEATURES + [
        "daytype",
        "weekday_hour",
        "daytype_time_period",
        "month_time_period",
        "season_time_period",
        "rideable_time_period",
        "rideable_daytype",
        "trip_direction_bin",
        "move_type",
        "same_grid",
        "route_station_id",
        "rideable_duration_bin",
        "rideable_distance_bin",
        "weekend_duration_bin",
        "time_direction",
        "same_grid_time",
    ]:
        df[col] = df[col].astype("string").fillna("missing")

    all_feature_columns = [col for col in df.columns if col != TARGET]
    X = df[all_feature_columns].copy()
    y = df[TARGET].astype(str).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    broad_grid = {
        "model__C": [0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0],
        "model__class_weight": [None, "balanced"],
    }
    fast_grid = {"model__C": [0.1, 0.25, 0.5, 1.0, 2.0]}

    experiments = [
        Experiment(
            name="baseline_current_features",
            param_grid=broad_grid,
            notes="current feature set, wider C and class_weight grid",
        ),
        Experiment(
            name="time_interactions",
            categorical=[
                "daytype",
                "weekday_hour",
                "daytype_time_period",
                "month_time_period",
                "season_time_period",
                "rideable_time_period",
                "rideable_daytype",
            ],
            param_grid=fast_grid,
            notes="PLAN time interaction feature set",
        ),
        Experiment(
            name="direction_numeric",
            numeric=["abs_lat_delta", "abs_lng_delta", "trip_angle_sin", "trip_angle_cos"],
            categorical=["trip_direction_bin", "move_type"],
            param_grid=fast_grid,
            notes="trip direction and displacement features",
        ),
        Experiment(
            name="numeric_transforms",
            numeric=[
                "duration_log1p",
                "distance_log1p",
                "speed_log1p",
                "duration_per_km",
                "distance_per_min",
            ],
            param_grid=fast_grid,
            notes="log and ratio features",
        ),
        Experiment(
            name="center_distance",
            use_center=True,
            param_grid=fast_grid,
            notes="data center distance features fitted only on train folds",
        ),
        Experiment(
            name="frequency_station_grid_route",
            frequency=[
                "start_station_id",
                "end_station_id",
                "route_station_id",
                "start_grid",
                "end_grid",
                "route_grid",
                "weekday_hour",
            ],
            param_grid=fast_grid,
            notes="unsupervised frequency features fitted only on train folds",
        ),
        Experiment(
            name="target_mean_station_grid_route_time",
            target_mean=[
                "start_station_id",
                "end_station_id",
                "route_station_id",
                "start_grid",
                "end_grid",
                "route_grid",
                "weekday_hour",
                "time_period",
                "rideable_type",
            ],
            param_grid=fast_grid,
            notes="OOF target mean features with smoothing=20",
        ),
        Experiment(
            name="combined_safe_features",
            numeric=[
                "abs_lat_delta",
                "abs_lng_delta",
                "trip_angle_sin",
                "trip_angle_cos",
                "duration_log1p",
                "distance_log1p",
                "speed_log1p",
                "duration_per_km",
                "distance_per_min",
            ],
            categorical=[
                "daytype",
                "weekday_hour",
                "daytype_time_period",
                "month_time_period",
                "season_time_period",
                "rideable_time_period",
                "rideable_daytype",
                "trip_direction_bin",
                "move_type",
                "rideable_duration_bin",
                "rideable_distance_bin",
                "weekend_duration_bin",
                "time_direction",
                "same_grid_time",
            ],
            frequency=[
                "start_station_id",
                "end_station_id",
                "route_station_id",
                "start_grid",
                "end_grid",
                "route_grid",
                "weekday_hour",
            ],
            target_mean=[
                "start_station_id",
                "end_station_id",
                "route_station_id",
                "start_grid",
                "end_grid",
                "route_grid",
                "weekday_hour",
                "time_period",
                "rideable_type",
            ],
            use_center=True,
            min_frequency=20,
            param_grid=fast_grid,
            notes="combined non-leaky engineered features",
        ),
    ]

    rows: list[dict[str, object]] = []
    for idx, exp in enumerate(experiments, start=1):
        print(f"[{idx}/{len(experiments)}] {exp.name}", flush=True)
        row = run_experiment(exp, X_train, X_test, y_train, y_test)
        rows.append(row)
        pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).to_csv(
            tables_dir / "optimization_experiments.csv",
            index=False,
        )
        print(
            f"  weighted_f1={row['weighted_f1']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} casual_recall={row['casual_recall']:.4f}",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    result.to_csv(tables_dir / "optimization_experiments.csv", index=False)
    print(result[["experiment_name", "weighted_f1", "macro_f1", "casual_recall", "casual_f1", "best_params"]].to_string(index=False))


if __name__ == "__main__":
    main()
