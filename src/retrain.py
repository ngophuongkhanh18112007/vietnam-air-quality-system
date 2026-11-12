
import os
import gc
import json
import argparse
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score
)

from config import (
    HF_TOKEN,
    DATASET_REPO_ID,
    MODEL_REPO_ID,
    DATASET_FILENAME,
    MODEL_FILENAME,
    FEATURE_COLUMNS,
    TOTAL_CITIES
)

from provinces import CITIES_63


# ============================================================
# 1. CONFIGURATION
# ============================================================

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide"
]

THRESHOLDS = {
    "pm2_5": 15.0,
    "pm10": 45.0,
    "carbon_monoxide": 4000.0,
    "nitrogen_dioxide": 25.0
}

LAGS = [1, 3, 6, 24]
ROLLING_WINDOWS = [3, 6]

CAT_FEATURES = ["city", "region"]

RAW_COLUMNS = [
    "time",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "city",
    "region"
]

LAST_RUN_FILE = "retraining/last_run.json"
RUNS_DIR = "retraining/runs"

# Candidate must not be worse than production on Macro F1.
# Class 3 recall may decrease by at most 0.02.
MAX_CLASS3_RECALL_DROP = 0.02

# Retraining interval
RETRAIN_INTERVAL_HOURS = 72


# ============================================================
# 2. CHECK RETRAINING SCHEDULE
# ============================================================

def should_retrain(force=False):

    if force:
        print("Manual retraining: bỏ qua kiểm tra chu kỳ.")
        return True

    print("Kiểm tra thời điểm retraining gần nhất...")

    try:
        last_run_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=LAST_RUN_FILE,
            repo_type="model",
            token=HF_TOKEN
        )

        with open(last_run_path, "r", encoding="utf-8") as f:
            last_run = json.load(f)

        last_time = datetime.fromisoformat(
            last_run["completed_at"]
        )

        now = datetime.now(timezone.utc)

        elapsed = now - last_time
        elapsed_hours = elapsed.total_seconds() / 3600

        print(f"Lần retraining gần nhất: {last_time}")
        print(f"Đã trôi qua: {elapsed_hours:.2f} giờ")

        if elapsed < timedelta(hours=RETRAIN_INTERVAL_HOURS):
            print("Chưa đủ 72 giờ. Bỏ qua lần chạy này.")
            return False

    except Exception as e:
        print(
            "Chưa tìm thấy lịch sử retraining "
            "hoặc không đọc được file:",
            str(e)
        )
        print("Tiếp tục retraining.")

    return True


# ============================================================
# 3. DOWNLOAD HISTORICAL DATA
# ============================================================

def download_historical_data():

    print("\n" + "=" * 65)
    print("TẢI DỮ LIỆU LỊCH SỬ TỪ HUGGING FACE")
    print("=" * 65)

    frames = []

    # 3.1. Download full baseline dataset
    print("\nĐang tải dataset nền:", DATASET_FILENAME)

    baseline_path = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=DATASET_FILENAME,
        repo_type="dataset",
        token=HF_TOKEN
    )

    baseline = pd.read_csv(
        baseline_path,
        usecols=lambda col: col in RAW_COLUMNS
    )

    missing = [
        col for col in RAW_COLUMNS
        if col not in baseline.columns
    ]

    if missing:
        raise ValueError(
            f"Dataset nền thiếu các cột: {missing}"
        )

    frames.append(baseline[RAW_COLUMNS].copy())

    print("Số dòng dataset nền:", len(baseline))

    del baseline
    gc.collect()

    # 3.2. Find monthly historical files
    repo_files = list_repo_files(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN
    )

    monthly_files = sorted([
        filename
        for filename in repo_files
        if filename.startswith("historical/")
        and filename.endswith(".csv")
    ])

    print("\nSố file lịch sử theo tháng:", len(monthly_files))

    for filename in monthly_files:

        print("Đang tải:", filename)

        file_path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=filename,
            repo_type="dataset",
            token=HF_TOKEN
        )

        monthly_df = pd.read_csv(
            file_path,
            usecols=lambda col: col in RAW_COLUMNS
        )

        missing = [
            col for col in RAW_COLUMNS
            if col not in monthly_df.columns
        ]

        if missing:
            raise ValueError(
                f"{filename} thiếu các cột: {missing}"
            )

        frames.append(
            monthly_df[RAW_COLUMNS].copy()
        )

        print("Số dòng:", len(monthly_df))

        del monthly_df
        gc.collect()

    # 3.3. Merge all sources
    print("\nĐang hợp nhất dữ liệu...")

    df = pd.concat(
        frames,
        ignore_index=True
    )

    del frames
    gc.collect()

    # 3.4. Clean data
    df["time"] = pd.to_datetime(
        df["time"],
        errors="coerce"
    )

    for col in POLLUTANTS:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df["city"] = df["city"].astype("string").str.strip()
    df["region"] = df["region"].astype("string").str.strip()

    df = df.dropna(
        subset=["time", "city", "region"]
    ).copy()

    df["city"] = df["city"].astype(str)
    df["region"] = df["region"].astype(str)

    # 3.5. Remove duplicated city + time
    before = len(df)

    df = (
        df.sort_values(["city", "time"])
        .drop_duplicates(
            subset=["city", "time"],
            keep="last"
        )
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )

    print("Số bản ghi trùng đã loại:", before - len(df))
    print("Tổng số dòng sau hợp nhất:", len(df))
    print("Số địa điểm:", df["city"].nunique())

    # 3.6. Validate 63 locations
    expected_cities = {
        info["name"]
        for info in CITIES_63.values()
    }

    actual_cities = set(df["city"].unique())

    if (
        len(actual_cities) != TOTAL_CITIES
        or actual_cities != expected_cities
    ):
        raise ValueError(
            "Dữ liệu không đủ 63 địa điểm. "
            f"Thiếu: {expected_cities - actual_cities}"
        )

    expected_regions = {
        info["name"]: info["region"]
        for info in CITIES_63.values()
    }

    invalid_region = df[
        df.apply(
            lambda row: expected_regions.get(row["city"])
            != row["region"],
            axis=1
        )
    ]

    if not invalid_region.empty:
        raise ValueError(
            "Có địa điểm không khớp region."
        )

    # 3.7. Check hourly continuity
    time_diff = (
        df.groupby("city")["time"]
        .diff()
    )

    irregular = time_diff.notna() & (
        time_diff != pd.Timedelta(hours=1)
    )

    if irregular.any():
        raise ValueError(
            "Dữ liệu không liên tục theo giờ. "
            "Không thể tạo target t+3 chính xác."
        )

    print("Dữ liệu liên tục theo giờ: OK")
    print("Khoảng thời gian:", df["time"].min(), "->", df["time"].max())

    return df


# ============================================================
# 4. CREATE RISK LABELS
# ============================================================

def create_risk_labels(df):

    print("\nĐang tính lại nhãn rủi ro...")

    df = df.copy()

    for col in POLLUTANTS:

        df[f"{col}_ratio"] = np.nan

    # Calculate rolling 24-hour mean separately by city
    for city, group_indices in df.groupby("city").groups.items():

        group = (
            df.loc[group_indices]
            .sort_values("time")
            .set_index("time")
        )

        ratios = []

        for col in POLLUTANTS:

            rolling_mean = group[col].rolling(
                "24h",
                min_periods=18
            ).mean()

            ratio = rolling_mean / THRESHOLDS[col]

            ratios.append(ratio)

        ratio_df = pd.concat(
            ratios,
            axis=1
        )

        # Require all four ratios to be available
        pollution_score = ratio_df.max(
            axis=1,
            skipna=False
        )

        labels = np.select(
            [
                pollution_score <= 1,
                (pollution_score > 1) & (pollution_score <= 2),
                (pollution_score > 2) & (pollution_score <= 3),
                pollution_score > 3
            ],
            [0, 1, 2, 3],
            default=np.nan
        )

        df.loc[group.index, "pollution_score"] = (
            pollution_score.to_numpy()
        )

        df.loc[group.index, "asthma_risk_level"] = labels

    print(
        "Phân bố nhãn:\n",
        df["asthma_risk_level"].value_counts(
            dropna=False
        ).sort_index()
    )

    return df


# ============================================================
# 5. CREATE 33 FEATURES AND TARGET t+3
# ============================================================

def create_training_dataset(df):

    print("\nĐang tạo 33 features và target t+3...")

    df = df.sort_values(
        ["city", "time"]
    ).copy()

    # 5.1. Lag features
    for col in POLLUTANTS:

        for lag in LAGS:

            feature_name = f"{col}_lag_{lag}h"

            df[feature_name] = (
                df.groupby("city")[col].shift(lag)
            )

    # 5.2. Rolling features
    for col in POLLUTANTS:

        for window in ROLLING_WINDOWS:

            feature_name = f"{col}_rolling_{window}h"

            df[feature_name] = (
                df.groupby("city")[col]
                .transform(
                    lambda s: s.rolling(
                        window=window,
                        min_periods=window
                    ).mean()
                )
            )

    # 5.3. Time features
    df["hour"] = df["time"].dt.hour
    df["day_of_week"] = df["time"].dt.dayofweek
    df["month"] = df["time"].dt.month

    # 5.4. Categorical features
    df["city"] = df["city"].astype(str)
    df["region"] = df["region"].astype(str)

    # 5.5. Target = risk level 3 hours later
    df["target"] = (
        df.groupby("city")["asthma_risk_level"].shift(-3)
    )

    df["target_time"] = (
        df["time"] + pd.Timedelta(hours=3)
    )

    # 5.6. Validate exact feature schema
    missing_features = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Thiếu features: {missing_features}"
        )

    if len(FEATURE_COLUMNS) != 33:
        raise ValueError(
            "FEATURE_COLUMNS phải có đúng 33 features."
        )

    # 5.7. Keep only valid training records
    required_columns = FEATURE_COLUMNS + ["target"]

    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    df["target"] = df["target"].astype(int)

    if not df["target"].isin([0, 1, 2, 3]).all():
        raise ValueError("Target có lớp không hợp lệ.")

    print("Số dòng ban đầu:", before)
    print("Số dòng dùng huấn luyện:", len(df))
    print("Số features:", len(FEATURE_COLUMNS))
    print("Các lớp target:", sorted(df["target"].unique()))

    return df


# ============================================================
# 6. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def temporal_split(df):

    print("\nĐang chia dữ liệu theo thời gian...")

    # Keep all 63 cities at the same target_time
    unique_times = np.sort(
        df["target_time"].unique()
    )

    n_times = len(unique_times)

    train_end = int(n_times * 0.80)
    val_end = int(n_times * 0.90)

    train_times = unique_times[:train_end]
    val_times = unique_times[train_end:val_end]
    test_times = unique_times[val_end:]

    train_df = df[
        df["target_time"].isin(train_times)
    ].copy()

    val_df = df[
        df["target_time"].isin(val_times)
    ].copy()

    test_df = df[
        df["target_time"].isin(test_times)
    ].copy()

    print("Train:", train_df.shape)
    print("Validation:", val_df.shape)
    print("Test:", test_df.shape)

    print(
        "Train target_time:",
        train_df["target_time"].min(),
        "->",
        train_df["target_time"].max()
    )

    print(
        "Validation target_time:",
        val_df["target_time"].min(),
        "->",
        val_df["target_time"].max()
    )

    print(
        "Test target_time:",
        test_df["target_time"].min(),
        "->",
        test_df["target_time"].max()
    )

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError("Một trong các tập dữ liệu bị rỗng.")

    return train_df, val_df, test_df


# ============================================================
# 7. EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = (
        np.asarray(model.predict(X_test))
        .reshape(-1)
        .astype(int)
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        average="weighted",
        zero_division=0
    )

    class3_recall = recall_score(
        y_test,
        predictions,
        labels=[3],
        average="macro",
        zero_division=0
    )

    report = classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3],
        target_names=[
            "Risk 0",
            "Risk 1",
            "Risk 2",
            "Risk 3"
        ],
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1, 2, 3]
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "class3_recall": float(class3_recall),
        "report": report,
        "confusion_matrix": cm,
        "predictions": predictions
    }


# ============================================================
# 8. TRAIN AND COMPARE MODELS
# ============================================================

def train_and_evaluate(train_df, val_df, test_df):

    print("\n" + "=" * 65)
    print("HUẤN LUYỆN CATBOOST CANDIDATE")
    print("=" * 65)

    X_train = train_df[FEATURE_COLUMNS].copy()
    y_train = train_df["target"].copy()

    X_val = val_df[FEATURE_COLUMNS].copy()
    y_val = val_df["target"].copy()

    X_test = test_df[FEATURE_COLUMNS].copy()
    y_test = test_df["target"].copy()

    # Ensure categorical values are strings
    for col in CAT_FEATURES:
        X_train[col] = X_train[col].astype(str)
        X_val[col] = X_val[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    # Calculate class weights from Train only
    class_counts = (
        y_train.value_counts()
        .reindex([0, 1, 2, 3], fill_value=0)
    )

    if (class_counts == 0).any():
        raise ValueError(
            f"Train thiếu lớp: {class_counts.to_dict()}"
        )

    class_weights = (
        len(y_train) /
        (4 * class_counts)
    ).tolist()

    print("Class weights:", class_weights)

    # Train candidate with original parameters
    candidate = CatBoostClassifier(
        iterations=1500,
        depth=8,
        learning_rate=0.05,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        class_weights=class_weights,
        l2_leaf_reg=5,
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
        allow_writing_files=False,
        thread_count=-1
    )

    candidate.fit(
        X_train,
        y_train,
        cat_features=CAT_FEATURES,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    # Check candidate feature schema
    if list(candidate.feature_names_) != FEATURE_COLUMNS:
        raise ValueError(
            "Candidate feature order không khớp config.py!"
        )

    if list(candidate.classes_) != [0, 1, 2, 3]:
        raise ValueError(
            f"Candidate classes không hợp lệ: {candidate.classes_}"
        )

    print("Best iteration:", candidate.best_iteration_)

    # Load current production model
    print("\nĐang tải production model...")

    production_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=HF_TOKEN
    )

    production = CatBoostClassifier()
    production.load_model(production_path)

    if list(production.feature_names_) != FEATURE_COLUMNS:
        raise ValueError(
            "Production model không khớp 33 features!"
        )

    if list(production.classes_) != [0, 1, 2, 3]:
        raise ValueError(
            "Production model không có đủ 4 lớp."
        )

    # Evaluate both models on exactly the same Test set
    print("\nĐang đánh giá production model...")
    production_metrics = evaluate_model(
        production,
        X_test,
        y_test
    )

    print("\nĐang đánh giá candidate model...")
    candidate_metrics = evaluate_model(
        candidate,
        X_test,
        y_test
    )

    print("\n========== SO SÁNH TEST ==========")

    print(
        "Production Macro F1:",
        round(production_metrics["macro_f1"], 4)
    )

    print(
        "Candidate Macro F1:",
        round(candidate_metrics["macro_f1"], 4)
    )

    print(
        "Production Class 3 Recall:",
        round(production_metrics["class3_recall"], 4)
    )

    print(
        "Candidate Class 3 Recall:",
        round(candidate_metrics["class3_recall"], 4)
    )

    # Promotion criteria
    macro_f1_pass = (
        candidate_metrics["macro_f1"]
        >= production_metrics["macro_f1"]
    )

    class3_pass = (
        candidate_metrics["class3_recall"]
        >= production_metrics["class3_recall"]
        - MAX_CLASS3_RECALL_DROP
    )

    promote = macro_f1_pass and class3_pass

    print("\n========== QUYẾT ĐỊNH ==========")

    if promote:
        print("CANDIDATE ĐẠT TIÊU CHÍ.")
        print("Cho phép cập nhật production model.")
    else:
        print("CANDIDATE CHƯA ĐẠT TIÊU CHÍ.")
        print("Giữ nguyên production model.")

    return (
        candidate,
        candidate_metrics,
        production_metrics,
        promote,
        X_test,
        y_test
    )


# ============================================================
# 9. SAVE AND UPLOAD RETRAINING RESULTS
# ============================================================

def upload_file(api, local_path, remote_path, commit_message):

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message=commit_message
    )


def save_and_upload_results(
    candidate,
    candidate_metrics,
    production_metrics,
    promote,
    test_df
):

    print("\nĐang lưu kết quả retraining...")

    run_id = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    api = HfApi(token=HF_TOKEN)

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(temp_dir)

        # 9.1. Metrics
        metrics = {
            "run_id": run_id,
            "completed_at": datetime.now(
                timezone.utc
            ).isoformat(),

            "data_start": str(test_df["time"].min()),
            "data_end": str(test_df["time"].max()),

            "train_rows": None,
            "validation_rows": None,
            "test_rows": len(test_df),

            "best_iteration": int(candidate.best_iteration_),

            "promotion": bool(promote),

            "production": {
                "accuracy": production_metrics["accuracy"],
                "macro_f1": production_metrics["macro_f1"],
                "weighted_f1": production_metrics["weighted_f1"],
                "class3_recall": production_metrics["class3_recall"]
            },

            "candidate": {
                "accuracy": candidate_metrics["accuracy"],
                "macro_f1": candidate_metrics["macro_f1"],
                "weighted_f1": candidate_metrics["weighted_f1"],
                "class3_recall": candidate_metrics["class3_recall"]
            },

            "feature_count": len(FEATURE_COLUMNS),
            "features": FEATURE_COLUMNS
        }

        metrics_path = temp_dir / "metrics.json"

        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(
                metrics,
                f,
                indent=4,
                ensure_ascii=False
            )

        # 9.2. Classification report
        report_path = temp_dir / "classification_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(candidate_metrics["report"])

        # 9.3. Confusion matrix
        cm_df = pd.DataFrame(
            candidate_metrics["confusion_matrix"],
            index=[
                "Actual_0",
                "Actual_1",
                "Actual_2",
                "Actual_3"
            ],
            columns=[
                "Predicted_0",
                "Predicted_1",
                "Predicted_2",
                "Predicted_3"
            ]
        )

        cm_path = temp_dir / "confusion_matrix.csv"
        cm_df.to_csv(cm_path)

        # 9.4. Feature importance
        importance_df = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": candidate.get_feature_importance()
        }).sort_values(
            "Importance",
            ascending=False
        )

        importance_path = temp_dir / "feature_importance.csv"
        importance_df.to_csv(
            importance_path,
            index=False
        )

        # Upload run artifacts first
        remote_prefix = f"{RUNS_DIR}/{run_id}"

        upload_file(
            api,
            metrics_path,
            f"{remote_prefix}/metrics.json",
            f"Save retraining metrics {run_id}"
        )

        upload_file(
            api,
            report_path,
            f"{remote_prefix}/classification_report.txt",
            f"Save classification report {run_id}"
        )

        upload_file(
            api,
            cm_path,
            f"{remote_prefix}/confusion_matrix.csv",
            f"Save confusion matrix {run_id}"
        )

        upload_file(
            api,
            importance_path,
            f"{remote_prefix}/feature_importance.csv",
            f"Save feature importance {run_id}"
        )

        # 9.5. Replace production model only if candidate passes
        if promote:

            candidate_path = temp_dir / "candidate.cbm"

            candidate.save_model(str(candidate_path))

            upload_file(
                api,
                candidate_path,
                MODEL_FILENAME,
                f"Promote CatBoost model {run_id}"
            )

            print("Đã cập nhật production model.")

        else:
            print("Không cập nhật production model.")

        # 9.6. Save last completed run for 72-hour gate
        last_run = {
            "run_id": run_id,
            "completed_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "promoted": bool(promote)
        }

        last_run_path = temp_dir / "last_run.json"

        with open(
            last_run_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                last_run,
                f,
                indent=4,
                ensure_ascii=False
            )

        upload_file(
            api,
            last_run_path,
            LAST_RUN_FILE,
            f"Update retraining schedule {run_id}"
        )

    print("Đã lưu kết quả retraining lên Hugging Face.")


# ============================================================
# 10. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--force",
        action="store_true",
        help="Chạy retraining ngay, bỏ qua chu kỳ 72 giờ."
    )

    args = parser.parse_args()

    print("=" * 65)
    print("VIETNAM AIR QUALITY - CATBOOST RETRAINING")
    print("=" * 65)

    if not HF_TOKEN:
        raise ValueError("Thiếu HF_TOKEN.")

    if not should_retrain(force=args.force):
        return

    # Download and merge historical data
    df = download_historical_data()

    # Recalculate risk labels
    df = create_risk_labels(df)

    # Create features and 3-hour future target
    training_df = create_training_dataset(df)

    del df
    gc.collect()

    # Temporal split
    train_df, val_df, test_df = temporal_split(training_df)

    # Train and compare
    (
        candidate,
        candidate_metrics,
        production_metrics,
        promote,
        X_test,
        y_test
    ) = train_and_evaluate(
        train_df,
        val_df,
        test_df
    )

    # Save and upload results
    save_and_upload_results(
        candidate,
        candidate_metrics,
        production_metrics,
        promote,
        test_df
    )

    print("\n" + "=" * 65)
    print("HOÀN THÀNH RETRAINING")
    print("=" * 65)

    print("Candidate Macro F1:", candidate_metrics["macro_f1"])
    print("Production Macro F1:", production_metrics["macro_f1"])
    print("Promoted:", promote)

    del training_df, train_df, val_df, test_df
    del X_test, y_test, candidate
    gc.collect()


if __name__ == "__main__":
    main()
