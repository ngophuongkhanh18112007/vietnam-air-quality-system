from concurrent.futures import ThreadPoolExecutor, as_completed
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
                timeout=90
            )

            # API giới hạn số request
            if response.status_code == 429:

                wait_time = int(
                    response.headers.get("Retry-After", 15)
                )

                print(
                    f"API giới hạn truy cập tại "
                    f"{city_info['name']}. "
                    f"Chờ {wait_time}s..."
                )

                if attempt < MAX_RETRIES:
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()

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

                wait_time = 5 * (2 ** (attempt - 1))

                print(
                    f"Thử lại sau {wait_time} giây..."
                )

                time.sleep(wait_time)

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
    print("Số luồng xử lý song song: 5")
    print("=" * 60)

    all_data = []
    failed_cities = []

    # Mỗi địa điểm có một nhiệm vụ riêng
    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = {}

        for city_id, city_info in CITIES_63.items():

            future = executor.submit(
                fetch_city_data,
                city_id,
                city_info,
                start_date,
                end_date
            )

            futures[future] = city_info

        # Nhận kết quả khi từng địa điểm hoàn thành
        completed = 0

        for future in as_completed(futures):

            completed += 1

            city_info = futures[future]

            print(
                f"\n[{completed}/{len(CITIES_63)}] "
                f"Hoàn thành: {city_info['name']}"
            )

            try:

                city_df = future.result()

                all_data.append(city_df)

                print(
                    f"  Thành công: {len(city_df)} dòng"
                )

            except Exception as error:

                failed_cities.append(
                    (city_info["name"], str(error))
                )

                print(f"  THẤT BẠI: {error}")

    # Kiểm tra đủ 63 địa điểm
    successful_cities = len(all_data)

    print("\n" + "=" * 60)
    print("KẾT QUẢ THU THẬP")
    print(f"Thành công: {successful_cities}/63")
    print(f"Thất bại: {len(failed_cities)}/63")
    print("=" * 60)

    # Không tiếp tục nếu thiếu địa điểm
    if failed_cities:

        print("\nCÁC ĐỊA ĐIỂM THẤT BẠI:")

        for city_name, error in failed_cities:
            print(f"- {city_name}: {error}")

        raise RuntimeError(
            f"Có {len(failed_cities)}/63 địa điểm thất bại. "
            "Không lưu dữ liệu chưa đầy đủ."
        )

    # Ghép dữ liệu sau khi đủ 63 địa điểm
    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    # Loại bỏ dữ liệu trùng
    final_df = final_df.drop_duplicates(
        subset=["city", "time"]
    )

    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    # Kiểm tra số lượng địa điểm
    actual_cities = final_df["city"].nunique()

    if actual_cities != 63:

        raise RuntimeError(
            f"Chỉ có {actual_cities}/63 địa điểm trong dữ liệu."
        )

    # Lưu dữ liệu khi đã kiểm tra thành công
    output_path = DATA_DIR / "latest_air_quality.csv"

    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTHU THẬP DỮ LIỆU THÀNH CÔNG!")
    print(f"Tổng số dòng: {len(final_df)}")
    print(f"Số địa điểm: {actual_cities}")
    print(f"Đường dẫn: {output_path}")

    return final_df

# 5. CHẠY CHƯƠNG TRÌNH
if __name__ == "__main__":
    fetch_all_cities()
