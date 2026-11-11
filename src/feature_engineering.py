import pandas as pd
import numpy as np

from datetime import datetime
from zoneinfo import ZoneInfo

from config import (
    DATA_DIR,
    TIMEZONE,
    FEATURE_COLUMNS,
    PREDICTION_HORIZON_HOURS
)

from provinces import CITIES_63


# 1. CẤU HÌNH

INPUT_FILE = DATA_DIR / "latest_air_quality.csv"

OUTPUT_FILE = DATA_DIR / "prediction_features.csv"

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide"
]

LAGS = [1, 3, 6, 24]

ROLLING_WINDOWS = [3, 6]


# 2. KIỂM TRA DỮ LIỆU ĐẦU VÀO

def load_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "time",
        "city",
        "region"
    ] + POLLUTANTS

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dữ liệu thiếu các cột: {missing}"
        )

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

    # Kiểm tra đủ 63 địa điểm
    expected_cities = {
        info["name"]
        for info in CITIES_63.values()
    }

    actual_cities = set(df["city"].unique())

    missing_cities = expected_cities - actual_cities

    if missing_cities or len(actual_cities) != 63:
        raise ValueError(
            f"Dữ liệu không đủ 63 địa điểm. "
            f"Thiếu: {missing_cities}"
        )

    # Kiểm tra bản ghi trùng
    if df.duplicated(["city", "time"]).any():
        raise ValueError(
            "Phát hiện bản ghi trùng city + time!"
        )

    # Kiểm tra region có khớp với cấu hình gốc
    expected_regions = {
        info["name"]: info["region"]
        for info in CITIES_63.values()
    }

    for city, region in zip(df["city"], df["region"]):

        if expected_regions.get(city) != region:
            raise ValueError(
                f"Region không khớp tại {city}: {region}"
            )

    return df

# 3. LOẠI BỎ DỮ LIỆU TƯƠNG LAI

def keep_available_data(df):

    now = datetime.now(ZoneInfo(TIMEZONE))

    # API trả về thời gian theo múi giờ Việt Nam.
    # Chuyển now về dạng datetime không có timezone
    # để so sánh với cột time.
    now_local = now.replace(tzinfo=None)

    df = df[
        df["time"] <= now_local
    ].copy()

    if df.empty:
        raise ValueError(
            "Không có dữ liệu tại hoặc trước thời điểm hiện tại!"
        )

    print(
        "Thời điểm hiện tại:",
        now_local
    )

    print(
        "Mốc dữ liệu mới nhất:",
        df["time"].max()
    )

    return df

# 4. KIỂM TRA TÍNH LIÊN TỤC THEO GIỜ

def check_hourly_data(df):

    df = df.sort_values(
        ["city", "time"]
    ).copy()

    time_diff = df.groupby("city")["time"].diff()

    irregular = time_diff.notna() & (
        time_diff != pd.Timedelta(hours=1)
    )

    if irregular.any():

        examples = df.loc[
            irregular,
            ["city", "time"]
        ].head(10)

        raise ValueError(
            "Dữ liệu không liên tục theo giờ!\n"
            f"Các mốc bất thường:\n{examples}"
        )

    print("Dữ liệu liên tục theo giờ!")

# 5. TẠO 33 FEATURES

def create_features(df):

    df = df.sort_values(
        ["city", "time"]
    ).copy()

    # 5.1. Lag Features
    for col in POLLUTANTS:

        for lag in LAGS:

            feature_name = f"{col}_lag_{lag}h"

            df[feature_name] = (
                df.groupby("city")[col].shift(lag)
            )

    # 5.2. Rolling Mean Features
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

    # 5.3. Time Features
    df["hour"] = df["time"].dt.hour

    df["day_of_week"] = df["time"].dt.dayofweek

    df["month"] = df["time"].dt.month

    # 5.4. Categorical Features
    df["city"] = df["city"].astype(str)

    df["region"] = df["region"].astype(str)

    # Kiểm tra đúng 33 features
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
            "Số lượng features trong config không phải 33!"
        )

    return df

# 6. CHỌN BẢN GHI MỚI NHẤT CỦA MỖI ĐỊA ĐIỂM

def select_latest_features(df):

    # Loại các dòng chưa đủ dữ liệu tạo features
    df = df.dropna(
        subset=FEATURE_COLUMNS
    ).copy()

    # Lấy dòng mới nhất của từng địa điểm
    latest_df = (
        df.sort_values(["city", "time"])
        .groupby("city", as_index=False)
        .tail(1)
        .copy()
    )

    # Kiểm tra đủ 63 địa điểm
    if latest_df["city"].nunique() != 63:
        missing = (
            set(info["name"] for info in CITIES_63.values())
            - set(latest_df["city"].unique())
        )

        raise ValueError(
            f"Không đủ 63 địa điểm có features hợp lệ. "
            f"Thiếu: {missing}"
        )

    # Thời điểm dự báo = thời điểm đầu vào + 3 giờ
    latest_df["prediction_time"] = (
        latest_df["time"]
        + pd.Timedelta(
            hours=PREDICTION_HORIZON_HOURS
        )
    )

    # Chỉ giữ các cột cần thiết
    output_columns = (
        ["time", "prediction_time"]
        + FEATURE_COLUMNS
    )

    latest_df = latest_df[output_columns].copy()

    latest_df = latest_df.sort_values(
        "city"
    ).reset_index(drop=True)

    return latest_df

# 7. CHƯƠNG TRÌNH CHÍNH

def run_feature_engineering():

    print("=" * 60)
    print("BẮT ĐẦU FEATURE ENGINEERING")
    print("=" * 60)

    # Đọc dữ liệu
    df = load_data()

    print("Số dòng ban đầu:", len(df))

    # Chỉ dùng dữ liệu đã có tại thời điểm hiện tại
    df = keep_available_data(df)

    # Kiểm tra thời gian
    check_hourly_data(df)

    # Tạo 33 features
    df = create_features(df)

    print("Đã tạo:", len(FEATURE_COLUMNS), "features")

    # Chọn dữ liệu dự báo
    prediction_df = select_latest_features(df)

    # Lưu kết quả
    prediction_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH FEATURE ENGINEERING")
    print("=" * 60)

    print("Số địa điểm:", prediction_df["city"].nunique())

    print("Số dòng đầu vào dự báo:", len(prediction_df))

    print("Số features:", len(FEATURE_COLUMNS))

    print(
        "Thời điểm dữ liệu:",
        prediction_df["time"].min(),
        "đến",
        prediction_df["time"].max()
    )

    print(
        "Thời điểm dự báo:",
        prediction_df["prediction_time"].min(),
        "đến",
        prediction_df["prediction_time"].max()
    )

    print("File kết quả:", OUTPUT_FILE)

    return prediction_df


if __name__ == "__main__":
    run_feature_engineering()
