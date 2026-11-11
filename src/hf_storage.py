from io import BytesIO

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from config import (
    HF_TOKEN,
    DATASET_REPO_ID,
    DATA_DIR
)

from provinces import CITIES_63


# 1. KHỞI TẠO HUGGING FACE API

api = HfApi(token=HF_TOKEN)

LOCAL_FILE = DATA_DIR / "latest_air_quality.csv"

REMOTE_FILE = "realtime/latest_air_quality.csv"

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


# 2. KIỂM TRA DỮ LIỆU HOÀN CHỈNH

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


# 3. LƯU CHECKPOINT CỦA MỘT NHÓM

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

    # Đường dẫn riêng cho từng workflow run và từng nhóm
    remote_checkpoint = (
        f"checkpoints/{run_id}/group_{group_id:02d}.csv"
    )

    # Chuyển DataFrame thành dữ liệu CSV trong bộ nhớ
    csv_bytes = df.to_csv(
        index=False,
        encoding="utf-8-sig"
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


# 4. ĐỌC CHECKPOINT ĐÃ LƯU

def load_checkpoint(run_id, group_id):

    remote_checkpoint = (
        f"checkpoints/{run_id}/group_{group_id:02d}.csv"
    )

    print(
        f"Kiểm tra checkpoint nhóm {group_id}..."
    )

    # Nếu checkpoint chưa tồn tại, trả về DataFrame rỗng
    if not api.file_exists(
        repo_id=DATASET_REPO_ID,
        filename=remote_checkpoint,
        repo_type="dataset"
    ):

        print(
            f"Chưa có checkpoint nhóm {group_id}."
        )

        return pd.DataFrame(columns=CHECKPOINT_COLUMNS)

    # Tải checkpoint từ Hugging Face
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


# 5. UPLOAD DỮ LIỆU CHÍNH LÊN HUGGING FACE

def upload_latest_data(df):

    validate_data(df)

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {LOCAL_FILE}"
        )

    print("\nĐang upload dữ liệu lên Hugging Face...")

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


# 6. CHẠY ĐỘC LẬP ĐỂ KIỂM TRA UPLOAD DỮ LIỆU CHÍNH

if __name__ == "__main__":

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có latest_air_quality.csv. "
            "Hãy chạy fetch_data.py trước."
        )

    df = pd.read_csv(LOCAL_FILE)

    upload_latest_data(df)
