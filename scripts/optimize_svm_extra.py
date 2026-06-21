#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from optimize_svm_experiments import (  # noqa: E402
    BASE_CATEGORICAL_FEATURES,
    BASE_NUMERIC_FEATURES,
    CLASS_ORDER,
    RANDOM_STATE,
    TARGET,
    Experiment,
    add_candidate_features,
    make_pipeline,
)


def make_grid_id(lat: pd.Series, lng: pd.Series, grid_size: float) -> pd.Series:
    lat_num = pd.to_numeric(lat, errors="coerce")
    lng_num = pd.to_numeric(lng, errors="coerce")
    lat_bin = np.floor(lat_num / grid_size).astype("Int64")
    lng_bin = np.floor(lng_num / grid_size).astype("Int64")
    return lat_bin.astype("string") + "_" + lng_bin.astype("string")


def add_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    out = add_candidate_features(df)
    for size in [0.003, 0.005, 0.008, 0.015, 0.02]:
        suffix = str(size).replace(".", "p")
        start_col = f"start_grid_{suffix}"
        end_col = f"end_grid_{suffix}"
        route_col = f"route_grid_{suffix}"
        out[start_col] = make_grid_id(out["start_lat"], out["start_lng"], size)
        out[end_col] = make_grid_id(out["end_lat"], out["end_lng"], size)
        out[route_col] = out[start_col].astype(str) + "->" + out[end_col].astype(str)

    out["route_station_name"] = (
        out["start_station_name"].astype("string").fillna("missing").astype(str)
        + "->"
        + out["end_station_name"].astype("string").fillna("missing").astype(str)
    )
    return out


def sanitize_categories(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    candidate_cats = [
        *BASE_CATEGORICAL_FEATURES,
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
        "start_station_name",
        "end_station_name",
        "route_station_name",
    ]
    candidate_cats.extend([c for c in out.columns if c.startswith(("start_grid_0p", "end_grid_0p", "route_grid_0p"))])
    for col in candidate_cats:
        if col in out.columns:
            out[col] = out[col].astype("string").fillna("missing")
    return out


def evaluate_prediction(name: str, best_params: dict[str, object], y_test: pd.Series, y_pred: np.ndarray, train_seconds: float, test_seconds: float, notes: str) -> dict[str, object]:
    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "experiment_name": name,
        "model": "LinearSVC",
        "best_params": json.dumps(best_params, ensure_ascii=False),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "member_f1": float(report["member"]["f1-score"]),
        "casual_precision": float(report["casual"]["precision"]),
        "casual_recall": float(report["casual"]["recall"]),
        "casual_f1": float(report["casual"]["f1-score"]),
        "train_time": train_seconds,
        "test_time": test_seconds,
        "notes": notes,
    }


def run_grid(exp: Experiment, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> dict[str, object]:
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
    row = evaluate_prediction(
        exp.name,
        grid.best_params_,
        y_test,
        y_pred,
        train_seconds,
        test_seconds,
        exp.notes,
    )
    row["best_cv_weighted_f1"] = float(grid.best_score_)
    return row


def run_threshold(
    name: str,
    exp: Experiment,
    params: dict[str, object],
    scoring: str,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    estimator = make_pipeline(exp)
    estimator.set_params(**params)
    tuner = TunedThresholdClassifierCV(
        estimator=estimator,
        scoring=scoring,
        response_method="decision_function",
        thresholds=200,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        refit=True,
        n_jobs=1,
        random_state=RANDOM_STATE,
        store_cv_results=False,
    )
    start = time.perf_counter()
    tuner.fit(X_train, y_train)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    y_pred = tuner.predict(X_test)
    test_seconds = time.perf_counter() - start
    best_threshold = getattr(tuner, "best_threshold_", None)
    return evaluate_prediction(
        name,
        {**params, "threshold_scoring": scoring, "best_threshold": best_threshold},
        y_test,
        y_pred,
        train_seconds,
        test_seconds,
        f"threshold tuned on training folds using {scoring}",
    )


def main() -> None:
    output_dir = Path("outputs")
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(output_dir / "processed" / "cleaned_stratified_sample.csv")
    df = sanitize_categories(add_extra_features(df))
    X = df[[c for c in df.columns if c != TARGET]].copy()
    y = df[TARGET].astype(str).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    broad_grid = {
        "model__C": [0.02, 0.03, 0.05, 0.1, 0.25, 0.5],
        "model__class_weight": [None, "balanced"],
    }
    small_grid = {
        "model__C": [0.02, 0.03, 0.05, 0.1],
        "model__class_weight": [None, "balanced"],
    }

    combined_numeric = [
        "abs_lat_delta",
        "abs_lng_delta",
        "trip_angle_sin",
        "trip_angle_cos",
        "duration_log1p",
        "distance_log1p",
        "speed_log1p",
        "duration_per_km",
        "distance_per_min",
    ]
    combined_categorical = [
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
    ]
    freq_cols = [
        "start_station_id",
        "end_station_id",
        "route_station_id",
        "start_grid",
        "end_grid",
        "route_grid",
        "weekday_hour",
    ]
    target_cols = [
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

    experiments: list[Experiment] = []
    for min_freq in [None, 5, 10, 20, 30, 50, 100]:
        experiments.append(
            Experiment(
                name=f"baseline_minfreq_{min_freq}",
                min_frequency=min_freq,
                param_grid=small_grid,
                notes="baseline feature set with one-hot min_frequency search",
            )
        )

    for size in [0.003, 0.005, 0.008, 0.015, 0.02]:
        suffix = str(size).replace(".", "p")
        experiments.append(
            Experiment(
                name=f"additional_grid_{suffix}",
                categorical=[f"start_grid_{suffix}", f"end_grid_{suffix}", f"route_grid_{suffix}"],
                min_frequency=20,
                param_grid=small_grid,
                notes=f"baseline plus floor-based grid_size={size}",
            )
        )

    experiments.extend(
        [
            Experiment(
                name="station_names",
                categorical=["start_station_name", "end_station_name", "route_station_name"],
                min_frequency=20,
                param_grid=small_grid,
                notes="station names and route names in addition to station ids",
            ),
            Experiment(
                name="frequency_broad",
                frequency=freq_cols,
                param_grid=broad_grid,
                notes="frequency features with class_weight search",
            ),
            Experiment(
                name="target_mean_broad",
                target_mean=target_cols,
                param_grid=broad_grid,
                notes="OOF target means with class_weight search",
            ),
            Experiment(
                name="combined_broad",
                numeric=combined_numeric,
                categorical=combined_categorical,
                frequency=freq_cols,
                target_mean=target_cols,
                use_center=True,
                min_frequency=20,
                param_grid=broad_grid,
                notes="combined engineered features with class_weight search",
            ),
        ]
    )

    rows: list[dict[str, object]] = []
    output_path = tables_dir / "optimization_extra_experiments.csv"
    for idx, exp in enumerate(experiments, start=1):
        print(f"[grid {idx}/{len(experiments)}] {exp.name}", flush=True)
        row = run_grid(exp, X_train, X_test, y_train, y_test)
        rows.append(row)
        pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).to_csv(output_path, index=False)
        print(
            f"  weighted_f1={row['weighted_f1']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} casual_recall={row['casual_recall']:.4f}",
            flush=True,
        )

    threshold_experiments = [
        (
            "baseline_threshold_weighted_f1",
            Experiment(name="baseline_threshold", min_frequency=30),
            {"model__C": 0.03, "model__class_weight": None},
            "f1_weighted",
        ),
        (
            "baseline_threshold_macro_f1",
            Experiment(name="baseline_threshold", min_frequency=30),
            {"model__C": 0.03, "model__class_weight": None},
            "f1_macro",
        ),
        (
            "target_mean_threshold_weighted_f1",
            Experiment(name="target_mean_threshold", target_mean=target_cols, min_frequency=30),
            {"model__C": 0.03, "model__class_weight": None},
            "f1_weighted",
        ),
        (
            "combined_threshold_weighted_f1",
            Experiment(
                name="combined_threshold",
                numeric=combined_numeric,
                categorical=combined_categorical,
                frequency=freq_cols,
                target_mean=target_cols,
                use_center=True,
                min_frequency=20,
            ),
            {"model__C": 0.03, "model__class_weight": None},
            "f1_weighted",
        ),
    ]

    for idx, (name, exp, params, scoring) in enumerate(threshold_experiments, start=1):
        print(f"[threshold {idx}/{len(threshold_experiments)}] {name}", flush=True)
        row = run_threshold(name, exp, params, scoring, X_train, X_test, y_train, y_test)
        rows.append(row)
        pd.DataFrame(rows).sort_values("weighted_f1", ascending=False).to_csv(output_path, index=False)
        print(
            f"  weighted_f1={row['weighted_f1']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} casual_recall={row['casual_recall']:.4f}",
            flush=True,
        )

    result = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
    result.to_csv(output_path, index=False)
    print(result[["experiment_name", "weighted_f1", "macro_f1", "casual_recall", "casual_f1", "best_params"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
