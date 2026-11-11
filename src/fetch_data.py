import time
import requests
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from config import DATA_DIR, TIMEZONE
from provinces import CITIES_63


# 1. CẤU HÌNH OPEN-METEO

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "european_aqi",
    "us_aqi"
]

# Lấy dữ liệu 3 ngày gần nhất
LOOKBACK_DAYS = 3

MAX_RETRIES = 3
REQUEST_TIMEOUT = 60


# 2. TẠO KHOẢNG THỜI GIAN

def get_date_range():
    now = datetime.now(ZoneInfo(TIMEZONE))

    end_date = now.date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    return start_date.isoformat(), end_date.isoformat()


# 3. LẤY DỮ LIỆU MỘT ĐỊA ĐIỂM

def fetch_city_data(city_id, city_info, start_date, end_date):

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "hourly": ",".join(POLLUTANTS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            if response.status_code == 429:
                wait_time = int(
                    response.headers.get("Retry-After", 10)
                )
                print(f"API giới hạn truy cập. Chờ {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            result = response.json()

            if "hourly" not in result:
                raise ValueError(
                    f"API không trả về dữ liệu hourly: {city_id}"
                )

            hourly = result["hourly"]

            df = pd.DataFrame(hourly)

            required_columns = ["time"] + POLLUTANTS

            missing = [
                col for col in required_columns
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Thiếu cột {missing} tại {city_id}"
                )

            df["time"] = pd.to_datetime(df["time"])

            df["city"] = city_info["name"]
            df["region"] = city_info["region"]

            return df

        except Exception as error:

            print(
                f"Lần thử {attempt}/{MAX_RETRIES} "
                f"thất bại tại {city_info['name']}: {error}"
            )

            if attempt < MAX_RETRIES:
                time.sleep(5)

    raise RuntimeError(
        f"Không thể lấy dữ liệu tại {city_info['name']}"
    )


# 4. LẤY DỮ LIỆU CHO CẢ 63 ĐỊA ĐIỂM

def fetch_all_cities():

    start_date, end_date = get_date_range()

    print("=" * 60)
    print("BẮT ĐẦU THU THẬP DỮ LIỆU")
    print(f"Thời gian: {start_date} đến {end_date}")
    print(f"Số địa điểm: {len(CITIES_63)}")
    print("=" * 60)

    all_data = []
    failed_cities = []

    for index, (city_id, city_info) in enumerate(
        CITIES_63.items(), start=1
    ):

        print(
            f"[{index}/63] Đang lấy dữ liệu: "
            f"{city_info['name']}"
        )

        try:
            city_df = fetch_city_data(
                city_id,
                city_info,
                start_date,
                end_date
            )

            all_data.append(city_df)

            print(
                f"  Thành công: {len(city_df)} dòng"
            )

        except Exception as error:

            failed_cities.append(
                (city_info["name"], str(error))
            )

            print(f"  THẤT BẠI: {error}")

        # Tránh gửi request quá nhanh
        time.sleep(2.8)

    # Không tiếp tục nếu thiếu địa điểm
    if failed_cities:

        print("\nCÁC ĐỊA ĐIỂM THẤT BẠI:")

        for city_name, error in failed_cities:
            print(f"- {city_name}: {error}")

        raise RuntimeError(
            f"Có {len(failed_cities)}/63 địa điểm thất bại. "
            "Không lưu dữ liệu chưa đầy đủ."
        )

    # 5. GỘP DỮ LIỆU

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset=["city", "time"],
        keep="last"
    )

    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # 6. LƯU FILE

    output_path = DATA_DIR / "latest_air_quality.csv"

    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("HOÀN THÀNH THU THẬP DỮ LIỆU")
    print(f"Tổng số dòng: {len(final_df):,}")
    print(f"Số địa điểm: {final_df['city'].nunique()}")
    print(f"File lưu tại: {output_path}")
    print("=" * 60)

    return final_df

# 7. CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    fetch_all_cities()
