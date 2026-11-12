
from io import BytesIO

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from config import (
    HF_TOKEN,
    DATASET_REPO_ID,
    DATA_DIR
)

from provinces import CITIES_63


# ============================================================
# 1. KHOI TAO HUGGING FACE API
# ============================================================

api = HfApi(token=HF_TOKEN)

LOCAL_FILE = DATA_DIR / "latest_air_quality.csv"

REMOTE_FILE = "realtime/latest_air_quality.csv"

HISTORICAL_DIR = "historical"

CHECKPOINT_COLUMNS = [
    "time",
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "european_aqi",
    "us_aqi",
    "city",
    "region"
]


# ============================================================
# 2. KIEM TRA DU LIEU HOAN CHINH
# ============================================================

def validate_data(df):

    if df.empty:
        raise ValueError("Dữ liệu thu thập được đang trống!")

    actual_cities = set(df["city"].unique())

    expected_cities = {
        info["name"]
        for info in CITIES_63.values()
    }

    missing_cities = expected_cities - actual_cities

    if missing_cities:
        raise ValueError(
            f"Thiếu dữ liệu của các địa điểm: {missing_cities}"
        )

    if len(actual_cities) != 63:
        raise ValueError(
            f"Số địa điểm không đúng: {len(actual_cities)}"
        )

    missing_columns = [
        col for col in CHECKPOINT_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột: {missing_columns}"
        )

    print("Kiểm tra dữ liệu thành công!")
    print(f"Số địa điểm: {len(actual_cities)}")
    print(f"Số dòng: {len(df):,}")


# ============================================================
# 3. LUU CHECKPOINT CUA MOT NHOM
# ============================================================

def save_checkpoint(df, run_id, group_id):

    if df.empty:
        print(f"Nhóm {group_id}: Không có dữ liệu để lưu.")
        return

    missing_columns = [
        col for col in CHECKPOINT_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Checkpoint thiếu cột: {missing_columns}"
        )

    remote_checkpoint = (
        f"checkpoints/{run_id}/group_{group_id:02d}.csv"
    )

    csv_bytes = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    print(
        f"Đang lưu checkpoint nhóm {group_id} "
        f"({df['city'].nunique()} địa điểm)..."
    )

    api.upload_file(
        path_or_fileobj=BytesIO(csv_bytes),
        path_in_repo=remote_checkpoint,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message=(
            f"Save checkpoint run {run_id}, "
            f"group {group_id}"
        )
    )

    print(f"Đã lưu checkpoint: {remote_checkpoint}")


# ============================================================
# 4. DOC CHECKPOINT DA LUU
# ============================================================

def load_checkpoint(run_id, group_id):

    remote_checkpoint = (
        f"checkpoints/{run_id}/group_{group_id:02d}.csv"
    )

    print(f"Kiểm tra checkpoint nhóm {group_id}...")

    if not api.file_exists(
        repo_id=DATASET_REPO_ID,
        filename=remote_checkpoint,
        repo_type="dataset"
    ):

        print(f"Chưa có checkpoint nhóm {group_id}.")

        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)

    local_checkpoint = hf_hub_download(
        repo_id=DATASET_REPO_ID,
        filename=remote_checkpoint,
        repo_type="dataset",
        token=HF_TOKEN
    )

    df = pd.read_csv(local_checkpoint)

    missing_columns = [
        col for col in CHECKPOINT_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Checkpoint nhóm {group_id} "
            f"thiếu cột: {missing_columns}"
        )

    print(
        f"Đã khôi phục {df['city'].nunique()} địa điểm "
        f"từ checkpoint nhóm {group_id}."
    )

    return df


# ============================================================
# 5. LUU DU LIEU LICH SU THEO THANG
# ============================================================

def upload_historical_data(df):

    print("\n" + "=" * 55)
    print("BAT DAU CAP NHAT DU LIEU LICH SU")
    print("=" * 55)

    history_df = df[CHECKPOINT_COLUMNS].copy()

    # Chuan hoa thoi gian
    history_df["time"] = pd.to_datetime(
        history_df["time"],
        errors="coerce"
    )

    if history_df["time"].isna().any():
        raise ValueError(
            "Dữ liệu lịch sử có thời gian không hợp lệ!"
        )

    history_df["city"] = (
        history_df["city"].astype(str).str.strip()
    )

    history_df["region"] = (
        history_df["region"].astype(str).str.strip()
    )

    # Xac dinh thang cua tung ban ghi
    history_df["month"] = (
        history_df["time"].dt.strftime("%Y-%m")
    )

    # Tach du lieu theo thang
    for month, new_month_df in history_df.groupby(
        "month",
        sort=True
    ):

        remote_history_file = (
            f"{HISTORICAL_DIR}/{month}.csv"
        )

        new_month_df = new_month_df[
            CHECKPOINT_COLUMNS
        ].copy()

        print(f"\nĐang xử lý tháng {month}...")
        print(
            f"Số dòng dữ liệu mới: {len(new_month_df):,}"
        )

        # Neu file lich su da ton tai thi tai ve
        if api.file_exists(
            repo_id=DATASET_REPO_ID,
            filename=remote_history_file,
            repo_type="dataset"
        ):

            local_history_file = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=remote_history_file,
                repo_type="dataset",
                token=HF_TOKEN
            )

            old_month_df = pd.read_csv(
                local_history_file
            )

            missing_columns = [
                col for col in CHECKPOINT_COLUMNS
                if col not in old_month_df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"File {remote_history_file} "
                    f"thiếu cột: {missing_columns}"
                )

            old_month_df = old_month_df[
                CHECKPOINT_COLUMNS
            ].copy()

            old_month_df["time"] = pd.to_datetime(
                old_month_df["time"],
                errors="coerce"
            )

            if old_month_df["time"].isna().any():
                raise ValueError(
                    f"File lịch sử {month} có thời gian "
                    "không hợp lệ!"
                )

            # Kiem tra file co dung thang hay khong
            old_months = set(
                old_month_df["time"].dt.strftime("%Y-%m")
            )

            if old_months - {month}:
                raise ValueError(
                    f"File {remote_history_file} "
                    "chứa dữ liệu không thuộc tháng tương ứng!"
                )

            # Gop du lieu cu va moi
            merged_df = pd.concat(
                [old_month_df, new_month_df],
                ignore_index=True
            )

            print(
                f"Số dòng lịch sử trước cập nhật: "
                f"{len(old_month_df):,}"
            )

        else:

            print(
                f"Chưa có file lịch sử tháng {month}. "
                "Tạo file mới."
            )

            merged_df = new_month_df.copy()

        # Loai bo ban ghi trung
        # Neu trung city + time, uu tien du lieu moi
        merged_df = merged_df.drop_duplicates(
            subset=["city", "time"],
            keep="last"
        )

        # Sap xep du lieu
        merged_df = merged_df.sort_values(
            ["city", "time"]
        ).reset_index(drop=True)

        # Chuyen thanh CSV
        csv_bytes = merged_df.to_csv(
            index=False
        ).encode("utf-8-sig")

        # Upload len Hugging Face
        api.upload_file(
            path_or_fileobj=BytesIO(csv_bytes),
            path_in_repo=remote_history_file,
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            commit_message=(
                f"Update historical air quality data {month}"
            )
        )

        print(
            f"Đã cập nhật: {remote_history_file}"
        )

        print(
            f"Tổng số dòng sau cập nhật: "
            f"{len(merged_df):,}"
        )

        print(
            f"Số địa điểm: "
            f"{merged_df['city'].nunique()}"
        )

    print("\nHOAN THANH CAP NHAT DU LIEU LICH SU!")


# ============================================================
# 6. UPLOAD DU LIEU MOI NHAT LEN HUGGING FACE
# ============================================================

def upload_latest_data(df):

    validate_data(df)

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {LOCAL_FILE}"
        )

    # Cap nhat kho du lieu lich su truoc
    upload_historical_data(df)

    # Sau khi lich su duoc cap nhat thanh cong,
    # moi upload file realtime
    print("\nĐang upload dữ liệu mới nhất lên Hugging Face...")

    api.upload_file(
        path_or_fileobj=str(LOCAL_FILE),
        path_in_repo=REMOTE_FILE,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Update latest air quality data"
    )

    print("\nUPLOAD THÀNH CÔNG!")
    print(f"Repository: {DATASET_REPO_ID}")
    print(f"File: {REMOTE_FILE}")


# ============================================================
# 7. CHAY DOC LAP DE KIEM TRA UPLOAD DU LIEU
# ============================================================

if __name__ == "__main__":

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có latest_air_quality.csv. "
            "Hãy chạy fetch_data.py trước."
        )

    df = pd.read_csv(LOCAL_FILE)

    upload_latest_data(df)
