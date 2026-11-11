from huggingface_hub import HfApi

from config import (
    HF_TOKEN,
    DATASET_REPO_ID,
    DATA_DIR
)

from provinces import CITIES_63

# 1. KHỞI TẠO HUGGING FACE API

api = HfApi(token=HF_TOKEN)

LOCAL_FILE = DATA_DIR / "latest_air_quality.csv"

# File dữ liệu gần nhất trên Hugging Face
REMOTE_FILE = "realtime/latest_air_quality.csv"

# 2. KIỂM TRA DỮ LIỆU TRƯỚC KHI UPLOAD

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

    required_columns = [
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

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Thiếu các cột: {missing_columns}"
        )

    print("Kiểm tra dữ liệu thành công!")
    print(f"Số địa điểm: {len(actual_cities)}")
    print(f"Số dòng: {len(df):,}")

# 3. UPLOAD DỮ LIỆU LÊN HUGGING FACE

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


# 4. CHẠY ĐỘC LẬP ĐỂ KIỂM TRA

if __name__ == "__main__":

    import pandas as pd

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            "Chưa có latest_air_quality.csv. "
            "Hãy chạy fetch_data.py trước."
        )

    df = pd.read_csv(LOCAL_FILE)

    upload_latest_data(df)
