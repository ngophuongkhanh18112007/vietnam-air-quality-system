"""
Xác thực dự đoán: đối chiếu dự báo T+3 với dữ liệu thực tế khi đã có.

Luồng:
  1. Đọc lịch sử dự báo (predictions/history/*.csv) trên Hugging Face
  2. Đọc dữ liệu thực tế (historical/*.csv) và tính lại nhãn rủi ro
  3. Ghép theo (city, prediction_time) -> tính metric
  4. So sánh với baseline persistence
  5. Lưu và upload kết quả vào verification/
"""

import json
import argparse
from io import BytesIO

import numpy as np
import pandas as pd

from huggingface_hub import HfApi, hf_hub_download
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from config import HF_TOKEN, DATASET_REPO_ID, DATA_DIR
from retrain import create_risk_labels

LABELS = [0, 1, 2, 3]
HISTORY_PREFIX = "predictions/history/"
HISTORICAL_DIR = "historical"
REMOTE_DIR = "verification"
POLLUTANTS = ["pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide"]


# ============================================================
# 1. ĐỌC DỮ LIỆU
# ============================================================

def load_prediction_history(api):
    files = [
        f for f in api.list_repo_files(DATASET_REPO_ID, repo_type="dataset")
        if f.startswith(HISTORY_PREFIX) and f.endswith(".csv")
    ]

    if not files:
        raise FileNotFoundError(
            "Chưa có lịch sử dự báo. Hãy chạy predict.py (bản đã cập nhật) "
            "ít nhất vài giờ trước khi xác thực."
        )

    frames = []
    for f in sorted(files):
        path = hf_hub_download(
            repo_id=DATASET_REPO_ID, filename=f,
            repo_type="dataset", token=HF_TOKEN,
        )
        frames.append(pd.read_csv(path))

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df["prediction_time"] = pd.to_datetime(df["prediction_time"])
    df = df.drop_duplicates(["city", "time"], keep="last")
    return df


def load_actual_data(api, start, end):
    # Lùi 2 ngày để đủ cửa sổ 24h tính nhãn ở đầu kỳ
    months = pd.period_range(
        start - pd.Timedelta(days=2), end, freq="M"
    ).strftime("%Y-%m")

    frames = []
    for month in months:
        remote = f"{HISTORICAL_DIR}/{month}.csv"
        if not api.file_exists(
            repo_id=DATASET_REPO_ID, filename=remote, repo_type="dataset"
        ):
            continue
        path = hf_hub_download(
            repo_id=DATASET_REPO_ID, filename=remote,
            repo_type="dataset", token=HF_TOKEN,
        )
        frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("Không có dữ liệu thực tế trong kỳ xác thực.")

    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in POLLUTANTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = (
        df.dropna(subset=["time", "city"])
        .drop_duplicates(["city", "time"], keep="last")
        .sort_values(["city", "time"])
        .reset_index(drop=True)
    )
    return df


# ============================================================
# 2. GHÉP DỰ BÁO VỚI THỰC TẾ
# ============================================================

def build_verification_table(preds, actual_df):
    labeled = create_risk_labels(actual_df)[
        ["city", "time", "asthma_risk_level"]
    ].dropna()

    actual = labeled.rename(columns={
        "time": "prediction_time",
        "asthma_risk_level": "actual_risk_level",
    })
    current = labeled.rename(columns={
        "asthma_risk_level": "current_risk_level",
    })

    merged = preds.merge(
        actual, on=["city", "prediction_time"], how="inner"
    ).merge(
        current, on=["city", "time"], how="left"
    )

    merged["actual_risk_level"] = merged["actual_risk_level"].astype(int)
    merged["predicted_risk_level"] = merged["predicted_risk_level"].astype(int)
    return merged


# ============================================================
# 3. TÍNH METRIC
# ============================================================

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )

    try:
        qwk = float(cohen_kappa_score(
            y_true, y_pred, labels=LABELS, weights="quadratic"
        ))
        if np.isnan(qwk):
            qwk = None
    except Exception:
        qwk = None

    high = y_true >= 2
    miss_high = (
        float((y_pred[high] <= 1).mean()) if high.any() else None
    )

    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=LABELS, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
        )),
        "within_1_accuracy": float((np.abs(y_true - y_pred) <= 1).mean()),
        "quadratic_weighted_kappa": qwk,
        # Tỷ lệ bỏ sót: thực tế mức Cao/Rất cao nhưng dự báo Thấp/Trung bình
        "high_risk_miss_rate": miss_high,
        "per_class": {
            str(l): {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, l in enumerate(LABELS)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=LABELS
        ).tolist(),
    }


def compute_all(table):
    model = compute_metrics(
        table["actual_risk_level"], table["predicted_risk_level"]
    )

    base_df = table.dropna(subset=["current_risk_level"])
    baseline = None
    if len(base_df):
        baseline = compute_metrics(
            base_df["actual_risk_level"], base_df["current_risk_level"]
        )

    # Các mẫu mức rủi ro THỰC SỰ thay đổi sau 3 giờ
    # (persistence luôn sai ở đây, mô hình mới thể hiện giá trị)
    changed = None
    if len(base_df):
        mask = base_df["actual_risk_level"] != base_df["current_risk_level"]
        if mask.any():
            sub = base_df[mask]
            changed = {
                "n_changed": int(mask.sum()),
                "changed_rate": float(mask.mean()),
                "model_accuracy_on_changed": float(
                    (sub["actual_risk_level"]
                     == sub["predicted_risk_level"]).mean()
                ),
            }

    return {
        "model": model,
        "persistence_baseline": baseline,
        "changed_samples": changed,
    }


def daily_metrics(table):
    d = table.copy()
    d["date"] = d["prediction_time"].dt.date
    rows = []
    for date, g in d.groupby("date"):
        rows.append({
            "date": str(date),
            "n": len(g),
            "accuracy": float(
                (g["actual_risk_level"] == g["predicted_risk_level"]).mean()
            ),
            "macro_f1": float(f1_score(
                g["actual_risk_level"], g["predicted_risk_level"],
                labels=LABELS, average="macro", zero_division=0
            )),
        })
    return pd.DataFrame(rows)


# ============================================================
# 4. LƯU VÀ UPLOAD
# ============================================================

def upload_bytes(api, data, remote_path, message):
    api.upload_file(
        path_or_fileobj=BytesIO(data),
        path_in_repo=remote_path,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=message,
    )


def print_summary(res):
    m, b, c = res["model"], res["persistence_baseline"], res["changed_samples"]

    print("\n" + "=" * 60)
    print("KẾT QUẢ XÁC THỰC DỰ BÁO T+3")
    print("=" * 60)
    print(f"Số dự báo đã xác thực: {m['n']:,}")
    print(f"{'':22}{'Model':>10}{'Persistence':>14}")
    for key in ["accuracy", "macro_f1", "within_1_accuracy"]:
        bv = f"{b[key]:.4f}" if b else "n/a"
        print(f"{key:22}{m[key]:>10.4f}{bv:>14}")
    if m["high_risk_miss_rate"] is not None:
        bv = (f"{b['high_risk_miss_rate']:.4f}"
              if b and b["high_risk_miss_rate"] is not None else "n/a")
        print(f"{'high_risk_miss_rate':22}"
              f"{m['high_risk_miss_rate']:>10.4f}{bv:>14}")
    if c:
        print(f"\nTỷ lệ mẫu đổi mức sau 3h: {c['changed_rate']:.2%} "
              f"({c['n_changed']:,} mẫu)")
        print(f"Model đoán đúng trên các mẫu đổi mức: "
              f"{c['model_accuracy_on_changed']:.2%} (persistence = 0%)")
    print("\nMa trận nhầm lẫn (hàng = thực tế, cột = dự báo):")
    print(pd.DataFrame(
        m["confusion_matrix"], index=LABELS, columns=LABELS
    ))


# ============================================================
# 5. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30,
                        help="Chỉ xác thực các dự báo trong N ngày gần nhất")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    api = HfApi(token=HF_TOKEN)

    preds = load_prediction_history(api)
    cutoff = preds["time"].max() - pd.Timedelta(days=args.days)
    preds = preds[preds["time"] >= cutoff]
    print(f"Số dự báo trong {args.days} ngày: {len(preds):,}")

    actual_df = load_actual_data(
        api, preds["time"].min(), preds["prediction_time"].max()
    )

    table = build_verification_table(preds, actual_df)
    pending = len(preds) - len(table)
    print(f"Đã có thực tế: {len(table):,} | Chưa đến hạn/thiếu: {pending:,}")

    if table.empty:
        print("Chưa có dự báo nào đến hạn xác thực.")
        return

    res = compute_all(table)
    res["pending_predictions"] = int(pending)
    res["period"] = {
        "from": str(table["prediction_time"].min()),
        "to": str(table["prediction_time"].max()),
    }

    print_summary(res)

    out_dir = DATA_DIR / "verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_bytes = json.dumps(res, ensure_ascii=False, indent=2).encode("utf-8")
    cm_bytes = pd.DataFrame(
        res["model"]["confusion_matrix"],
        index=[f"actual_{l}" for l in LABELS],
        columns=[f"pred_{l}" for l in LABELS],
    ).to_csv().encode("utf-8-sig")
    daily_bytes = daily_metrics(table).to_csv(index=False).encode("utf-8-sig")
    detail_bytes = table.to_csv(index=False).encode("utf-8-sig")

    (out_dir / "metrics.json").write_bytes(metrics_bytes)
    (out_dir / "confusion_matrix.csv").write_bytes(cm_bytes)
    (out_dir / "daily_metrics.csv").write_bytes(daily_bytes)
    (out_dir / "verified_predictions.csv").write_bytes(detail_bytes)
    print(f"\nĐã lưu kết quả tại: {out_dir}")

    if args.no_upload:
        return

    msg = "Update prediction verification"
    for name, data in [
        ("metrics.json", metrics_bytes),
        ("confusion_matrix.csv", cm_bytes),
        ("daily_metrics.csv", daily_bytes),
        ("verified_predictions.csv", detail_bytes),
    ]:
        upload_bytes(api, data, f"{REMOTE_DIR}/{name}", msg)
    print("Upload kết quả xác thực thành công!")


if __name__ == "__main__":
    main()
