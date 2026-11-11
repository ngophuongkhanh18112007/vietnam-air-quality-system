
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from config import DATA_DIR, TIMEZONE
from provinces import CITIES_63
from hf_storage import save_checkpoint, load_checkpoint


# ============================================================
# 1. CẤU HÌNH OPEN-METEO
# ============================================================

API_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

POLLUTANTS = [
    "pm2_5",
    "pm10",
    "carbon_monoxide",
    "nitrogen_dioxide",
    "european_aqi",
    "us_aqi"
]

LOOKBACK_DAYS = 3

MAX_RETRIES = 3
REQUEST_TIMEOUT = 90

# Mỗi nhóm có 7 địa điểm
GROUP_SIZE = 7

# Mỗi nhóm tối đa 5 yêu cầu chạy song song
MAX_WORKERS = 5

REQUIRED_COLUMNS = [
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
# 2. TẠO KHOẢNG THỜI GIAN
# ============================================================

def get_date_range():

    now = datetime.now(ZoneInfo(TIMEZONE))

    end_date = now.date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    return start_date.isoformat(), end_date.isoformat()


# ============================================================
# 3. TẠO MÃ CHECKPOINT CHO LẦN CHẠY
# ============================================================

def get_checkpoint_run_id(start_date, end_date):

    github_run_id = os.getenv("GITHUB_RUN_ID")

    if github_run_id:
        base_id = github_run_id
    else:
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{base_id}_{start_date}_{end_date}"


# ============================================================
# 4. LẤY DỮ LIỆU MỘT ĐỊA ĐIỂM
# ============================================================

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

            result = response.json()

            if "hourly" not in result:
                raise ValueError(
                    f"API không trả về dữ liệu hourly: {city_id}"
                )

            df = pd.DataFrame(result["hourly"])

            missing = [
                col for col in ["time"] + POLLUTANTS
                if col not in df.columns
            ]

            if missing:
                raise ValueError(
                    f"Thiếu cột {missing} tại {city_id}"
                )

            if df.empty:
                raise ValueError(
                    f"API trả về dữ liệu trống tại {city_id}"
                )

            df["time"] = pd.to_datetime(df["time"])

            df["city"] = city_info["name"]
            df["region"] = city_info["region"]

            return df[REQUIRED_COLUMNS]

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


# ============================================================
# 5. CHIA 63 ĐỊA ĐIỂM THÀNH CÁC NHÓM
# ============================================================

def create_groups():

    cities = list(CITIES_63.items())

    groups = []

    for i in range(0, len(cities), GROUP_SIZE):

        groups.append(
            cities[i:i + GROUP_SIZE]
        )

    return groups


# ============================================================
# 6. KIỂM TRA CHECKPOINT CỦA MỘT NHÓM
# ============================================================

def get_valid_checkpoint_cities(
    checkpoint_df,
    group_cities,
    start_date,
    end_date
):

    valid_data = []

    expected_names = {
        info["name"]
        for _, info in group_cities
    }

    if checkpoint_df.empty:
        return valid_data

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in checkpoint_df.columns
    ]

    if missing_columns:
        print(
            "Checkpoint thiếu cột, sẽ thu thập lại nhóm."
        )
        return valid_data

    checkpoint_df = checkpoint_df.copy()

    checkpoint_df["time"] = pd.to_datetime(
        checkpoint_df["time"],
        errors="coerce"
    )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    checkpoint_df = checkpoint_df[
        (checkpoint_df["time"] >= start)
        & (checkpoint_df["time"] < end)
    ]

    # Tạo danh sách các mốc giờ dự kiến
    expected_hours = set(
        pd.date_range(
            start=start,
            end=end,
            freq="h",
            inclusive="left"
        )
    )

    # Kiểm tra từng địa điểm trong nhóm
    for city_name in expected_names:

        city_df = checkpoint_df[
            checkpoint_df["city"] == city_name
        ].copy()

        if city_df.empty:
            continue

        if city_df["time"].isna().any():
            continue

        # Loại bỏ các mốc giờ trùng lặp
        city_df = city_df.drop_duplicates(
            subset=["time"]
        )

        actual_hours = set(city_df["time"])

        # Kiểm tra checkpoint có đủ mốc giờ không
        if not expected_hours.issubset(actual_hours):

            print(
                f"  Checkpoint chưa đầy đủ: {city_name}. "
                "Sẽ thu thập lại địa điểm này."
            )

            continue

        valid_data.append(
            city_df[REQUIRED_COLUMNS]
        )

        print(
            f"  Khôi phục checkpoint: {city_name} "
            f"({len(city_df)} dòng)"
        )

    return valid_data


# ============================================================
# 7. THU THẬP TỪNG NHÓM VÀ LƯU CHECKPOINT
# ============================================================

def fetch_one_group(
    group_cities,
    group_number,
    total_groups,
    start_date,
    end_date,
    checkpoint_run_id
):

    print("\n" + "=" * 60)

    print(
        f"BẮT ĐẦU NHÓM {group_number}/{total_groups}"
    )

    print("=" * 60)

    # Đọc checkpoint của nhóm hiện tại
    checkpoint_df = load_checkpoint(
        checkpoint_run_id,
        group_number
    )

    group_data = get_valid_checkpoint_cities(
        checkpoint_df,
        group_cities,
        start_date,
        end_date
    )

    # Xác định các địa điểm đã có dữ liệu hợp lệ
    completed_names = set()

    for df in group_data:
        completed_names.update(
            df["city"].unique()
        )

    # Xác định các địa điểm còn thiếu
    missing_cities = [
        (city_id, city_info)
        for city_id, city_info in group_cities
        if city_info["name"] not in completed_names
    ]

    print(
        f"\nNhóm {group_number}: "
        f"Đã có {len(completed_names)}/{len(group_cities)} địa điểm."
    )

    print(
        f"Cần thu thập thêm: {len(missing_cities)} địa điểm."
    )

    failed_cities = []

    # Chỉ thu thập những địa điểm chưa có checkpoint hợp lệ
    if missing_cities:

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = {}

            for city_id, city_info in missing_cities:

                future = executor.submit(
                    fetch_city_data,
                    city_id,
                    city_info,
                    start_date,
                    end_date
                )

                futures[future] = city_info

            for future in as_completed(futures):

                city_info = futures[future]

                try:

                    city_df = future.result()

                    group_data.append(city_df)

                    print(
                        f"  Thành công: {city_info['name']} "
                        f"({len(city_df)} dòng)"
                    )

                except Exception as error:

                    failed_cities.append(
                        (city_info["name"], str(error))
                    )

                    print(
                        f"  THẤT BẠI: {city_info['name']}: {error}"
                    )

    # Lưu tất cả dữ liệu thành công của nhóm
    if group_data:

        group_df = pd.concat(
            group_data,
            ignore_index=True
        )

        group_df = group_df.drop_duplicates(
            subset=["city", "time"]
        )

        save_checkpoint(
            group_df,
            checkpoint_run_id,
            group_number
        )

    # Đếm số địa điểm thực sự có dữ liệu
    completed_count = len({
        city
        for df in group_data
        for city in df["city"].unique()
    })

    print(
        f"\nKẾT THÚC NHÓM {group_number}: "
        f"{completed_count}/{len(group_cities)} địa điểm có dữ liệu, "
        f"{len(failed_cities)} địa điểm thất bại."
    )

    return group_data, failed_cities


# ============================================================
# 8. THU THẬP TOÀN BỘ 63 ĐỊA ĐIỂM
# ============================================================

def fetch_all_cities():

    start_date, end_date = get_date_range()

    groups = create_groups()

    checkpoint_run_id = get_checkpoint_run_id(
        start_date,
        end_date
    )

    print("=" * 60)

    print("BẮT ĐẦU THU THẬP DỮ LIỆU")

    print(f"Thời gian: {start_date} đến {end_date}")
    print(f"Tổng số địa điểm: {len(CITIES_63)}")
    print(f"Số nhóm: {len(groups)}")
    print(f"Số địa điểm mỗi nhóm: tối đa {GROUP_SIZE}")
    print(f"Số luồng song song: {MAX_WORKERS}")
    print(f"Checkpoint ID: {checkpoint_run_id}")

    print("=" * 60)

    all_data = []
    all_failed_cities = []

    # Xử lý lần lượt từng nhóm
    for group_number, group_cities in enumerate(
        groups,
        start=1
    ):

        group_data, failed_cities = fetch_one_group(
            group_cities,
            group_number,
            len(groups),
            start_date,
            end_date,
            checkpoint_run_id
        )

        all_data.extend(group_data)
        all_failed_cities.extend(failed_cities)

    # ========================================================
    # 9. TỔNG HỢP VÀ KIỂM TRA
    # ========================================================

    print("\n" + "=" * 60)

    print("TỔNG KẾT THU THẬP")

    print("=" * 60)

    if not all_data:

        raise RuntimeError(
            "Không thu thập được dữ liệu nào."
        )

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )

    final_df = final_df.drop_duplicates(
        subset=["city", "time"]
    )

    final_df = final_df.sort_values(
        ["city", "time"]
    ).reset_index(drop=True)

    actual_cities = set(
        final_df["city"].unique()
    )

    expected_cities = {
        info["name"]
        for info in CITIES_63.values()
    }

    missing_cities = expected_cities - actual_cities

    print(
        f"Số địa điểm thu thập được: "
        f"{len(actual_cities)}/63"
    )

    print(
        f"Tổng số dòng: {len(final_df):,}"
    )

    if missing_cities:

        print("\nCÁC ĐỊA ĐIỂM CÒN THIẾU:")

        for city_name in sorted(missing_cities):
            print(f"- {city_name}")

        raise RuntimeError(
            f"Thiếu dữ liệu {len(missing_cities)} địa điểm. "
            "Checkpoint của các nhóm thành công đã được lưu. "
            "Hãy chạy lại workflow để tiếp tục."
        )

    # ========================================================
    # 10. LƯU FILE CSV CHÍNH
    # ========================================================

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = DATA_DIR / "latest_air_quality.csv"

    final_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)

    print("THU THẬP DỮ LIỆU THÀNH CÔNG!")

    print(f"Tổng số dòng: {len(final_df):,}")
    print(f"Số địa điểm: {len(actual_cities)}/63")
    print(f"Đường dẫn: {output_path}")

    print("=" * 60)

    return final_df


# ============================================================
# 11. CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":

    fetch_all_cities()
