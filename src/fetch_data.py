
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
    "us_aqi",
]

REQUIRED_COLUMNS = [
    "time",
    *POLLUTANTS,
    "city",
    "region",
]

LOOKBACK_DAYS = 3

# Cấu hình retry và timeout
MAX_RETRIES = 5
REQUEST_TIMEOUT = 120

# Giảm số request đồng thời để hạn chế quá tải API
GROUP_SIZE = 7
MAX_WORKERS = 3

# Thời gian chờ khi request gặp lỗi:
# Lần 1: 10 giây
# Lần 2: 20 giây
# Lần 3: 40 giây
# Lần 4: 80 giây


# ============================================================
# 2. KIỂM TRA DANH SÁCH 63 ĐỊA ĐIỂM
# ============================================================

def validate_cities():

    if len(CITIES_63) != 63:
        raise ValueError(
            f"Danh sách địa điểm không đủ 63. "
            f"Hiện tại: {len(CITIES_63)}"
        )

    expected_regions = {
        "Mien_Bac": 25,
        "Mien_Trung": 19,
        "Mien_Nam": 19,
    }

    actual_regions = {}

    for city_id, city_info in CITIES_63.items():

        required_keys = ["name", "region", "lat", "lon"]

        for key in required_keys:
            if key not in city_info:
                raise ValueError(
                    f"Địa điểm {city_id} thiếu thông tin {key}"
                )

        region = city_info["region"]

        actual_regions[region] = (
            actual_regions.get(region, 0) + 1
        )

    if actual_regions != expected_regions:
        raise ValueError(
            f"Phân bố vùng không đúng: {actual_regions}"
        )

    print("Đã kiểm tra danh sách 63 địa điểm thành công!")
    print(f"Phân bố: {actual_regions}")


# ============================================================
# 3. XÁC ĐỊNH KHOẢNG THỜI GIAN THU THẬP
# ============================================================

def get_date_range():

    now = datetime.now(ZoneInfo(TIMEZONE))

    # Chỉ lấy đến giờ đã hoàn thành
    cutoff_time = now.replace(
        minute=0,
        second=0,
        microsecond=0
    ) - timedelta(hours=1)

    end_date = now.date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    return (
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
        cutoff_time.replace(tzinfo=None),
    )


# ============================================================
# 4. TẠO CHECKPOINT RUN ID
# ============================================================

def get_checkpoint_run_id(start_date, end_date):

    github_run_id = os.getenv("GITHUB_RUN_ID")

    if github_run_id:
        base_id = github_run_id
    else:
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return f"{base_id}_{start_date}_{end_date}"


# ============================================================
# 5. LẤY DỮ LIỆU TỪ MỘT ĐỊA ĐIỂM
# ============================================================

def fetch_city_data(
    city_id,
    city_info,
    start_date,
    end_date,
    cutoff_time
):

    city_name = city_info["name"]

    params = {
        "latitude": city_info["lat"],
        "longitude": city_info["lon"],
        "hourly": ",".join(POLLUTANTS),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"[{city_name}] "
                f"Đang lấy dữ liệu "
                f"(lần thử {attempt}/{MAX_RETRIES})..."
            )

            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT
            )

            # ------------------------------------------------
            # XỬ LÝ HTTP 429 - API GIỚI HẠN REQUEST
            # ------------------------------------------------

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After",
                    "15"
                )

                try:
                    wait_time = int(retry_after)
                except (TypeError, ValueError):
                    wait_time = 15

                if attempt < MAX_RETRIES:

                    print(
                        f"[{city_name}] API giới hạn truy cập. "
                        f"Chờ {wait_time} giây rồi thử lại..."
                    )

                    time.sleep(wait_time)
                    continue

                raise RuntimeError(
                    f"API tiếp tục giới hạn truy cập tại "
                    f"{city_name} sau {MAX_RETRIES} lần thử."
                )

            response.raise_for_status()

            result = response.json()

            if "hourly" not in result:
                raise ValueError(
                    f"API không trả về dữ liệu hourly cho {city_name}"
                )

            hourly_data = result["hourly"]

            df = pd.DataFrame(hourly_data)

            # Kiểm tra các cột cần thiết
            missing_columns = [
                col for col in ["time", *POLLUTANTS]
                if col not in df.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"{city_name} thiếu các cột: {missing_columns}"
                )

            if df.empty:
                raise ValueError(
                    f"Không có dữ liệu trả về cho {city_name}"
                )

            # Chuyển thời gian
            df["time"] = pd.to_datetime(
                df["time"],
                errors="coerce"
            )

            df = df.dropna(subset=["time"])

            # Chỉ giữ dữ liệu đến giờ đã hoàn thành
            df = df[
                df["time"] <= cutoff_time
            ].copy()

            if df.empty:
                raise ValueError(
                    f"{city_name} không có dữ liệu "
                    f"trước thời điểm {cutoff_time}"
                )

            # Gắn thông tin địa điểm
            df["city"] = city_name
            df["region"] = city_info["region"]

            # Sắp xếp và loại bỏ bản ghi trùng
            df = (
                df[REQUIRED_COLUMNS]
                .drop_duplicates(subset=["time"])
                .sort_values("time")
                .reset_index(drop=True)
            )

            print(
                f"[{city_name}] Thu thập thành công: "
                f"{len(df)} dòng dữ liệu."
            )

            return df

        except Exception as error:

            if attempt < MAX_RETRIES:

                wait_time = 10 * (2 ** (attempt - 1))

                print(
                    f"[{city_name}] Lỗi lần "
                    f"{attempt}/{MAX_RETRIES}: {error}"
                )

                print(
                    f"[{city_name}] Thử lại sau "
                    f"{wait_time} giây..."
                )

                time.sleep(wait_time)

            else:

                print(
                    f"[{city_name}] THẤT BẠI sau "
                    f"{MAX_RETRIES} lần thử: {error}"
                )

                raise RuntimeError(
                    f"Không thể thu thập dữ liệu tại {city_name}"
                ) from error


# ============================================================
# 6. KIỂM TRA CHECKPOINT ĐÃ LƯU
# ============================================================

def get_valid_checkpoint_cities(
    checkpoint_df,
    group_cities,
    start_date,
    cutoff_time
):

    valid_cities = set()

    if checkpoint_df is None or checkpoint_df.empty:
        return valid_cities

    # Kiểm tra cột
    if not set(REQUIRED_COLUMNS).issubset(
        checkpoint_df.columns
    ):
        print("Checkpoint thiếu cột cần thiết. Sẽ thu thập lại.")
        return valid_cities

    checkpoint_df = checkpoint_df.copy()

    checkpoint_df["time"] = pd.to_datetime(
        checkpoint_df["time"],
        errors="coerce"
    )

    checkpoint_df = checkpoint_df.dropna(
        subset=["time"]
    )

    checkpoint_df = checkpoint_df[
        (checkpoint_df["time"] >= pd.Timestamp(start_date))
        & (checkpoint_df["time"] <= cutoff_time)
    ]

    # Các mốc giờ cần có
    expected_hours = pd.date_range(
        start=pd.Timestamp(start_date),
        end=cutoff_time,
        freq="h"
    )

    expected_set = set(expected_hours)

    for city_id, city_info in group_cities:

        city_name = city_info["name"]

        city_df = checkpoint_df[
            checkpoint_df["city"] == city_name
        ]

        if city_df.empty:
            continue

        actual_hours = set(
            city_df["time"].drop_duplicates()
        )

        # Chỉ sử dụng checkpoint nếu đủ toàn bộ mốc giờ
        if expected_set.issubset(actual_hours):

            valid_cities.add(city_id)

            print(
                f"[Checkpoint] {city_name}: "
                f"Đã có đủ dữ liệu, bỏ qua thu thập."
            )

        else:

            missing_count = len(
                expected_set - actual_hours
            )

            print(
                f"[Checkpoint] {city_name}: "
                f"Thiếu {missing_count} mốc giờ. "
                f"Sẽ thu thập lại."
            )

    return valid_cities


# ============================================================
# 7. THU THẬP DỮ LIỆU THEO TỪNG NHÓM
# ============================================================

def fetch_all_cities():

    print("=" * 60)
    print("BẮT ĐẦU THU THẬP DỮ LIỆU")
    print("=" * 60)

    validate_cities()

    start_date, end_date, cutoff_time = get_date_range()

    checkpoint_run_id = get_checkpoint_run_id(
        start_date,
        end_date
    )

    print(f"Thời gian bắt đầu: {start_date}")
    print(f"Thời gian kết thúc: {end_date}")
    print(f"Giờ dữ liệu cuối cùng: {cutoff_time}")
    print(f"Checkpoint Run ID: {checkpoint_run_id}")

    all_city_data = []
    all_failed_cities = []

    cities = list(CITIES_63.items())

    # Chia 63 địa điểm thành từng nhóm
    groups = [
        cities[i:i + GROUP_SIZE]
        for i in range(0, len(cities), GROUP_SIZE)
    ]

    print(f"Tổng số nhóm: {len(groups)}")

    # --------------------------------------------------------
    # XỬ LÝ TỪNG NHÓM
    # --------------------------------------------------------

    for group_number, group_cities in enumerate(
        groups,
        start=1
    ):

        print("\n" + "=" * 60)
        print(
            f"NHÓM {group_number}/{len(groups)} "
            f"- {len(group_cities)} địa điểm"
        )
        print("=" * 60)

        # Tải checkpoint của nhóm hiện tại
        checkpoint_df = load_checkpoint(
            checkpoint_run_id,
            group_number
        )

        if checkpoint_df is None:
            checkpoint_df = pd.DataFrame(
                columns=REQUIRED_COLUMNS
            )

        # Kiểm tra địa điểm nào đã có đủ dữ liệu
        valid_checkpoint_cities = get_valid_checkpoint_cities(
            checkpoint_df,
            group_cities,
            start_date,
            cutoff_time
        )

        group_data = []

        # Chỉ giữ dữ liệu checkpoint hợp lệ
        if not checkpoint_df.empty:

            valid_city_names = {
                city_info["name"]
                for city_id, city_info in group_cities
                if city_id in valid_checkpoint_cities
            }

            valid_checkpoint_df = checkpoint_df[
                checkpoint_df["city"].isin(
                    valid_city_names
                )
            ].copy()

            if not valid_checkpoint_df.empty:
                group_data.append(valid_checkpoint_df)

        # Các địa điểm cần thu thập lại
        cities_to_fetch = [
            (city_id, city_info)
            for city_id, city_info in group_cities
            if city_id not in valid_checkpoint_cities
        ]

        print(
            f"Số địa điểm cần thu thập: "
            f"{len(cities_to_fetch)}"
        )

        failed_in_group = []

        # ----------------------------------------------------
        # CHẠY SONG SONG VỚI SỐ WORKER GIỚI HẠN
        # ----------------------------------------------------

        if cities_to_fetch:

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                future_to_city = {
                    executor.submit(
                        fetch_city_data,
                        city_id,
                        city_info,
                        start_date,
                        end_date,
                        cutoff_time
                    ): (city_id, city_info)

                    for city_id, city_info in cities_to_fetch
                }

                for future in as_completed(future_to_city):

                    city_id, city_info = future_to_city[future]

                    try:

                        city_df = future.result()

                        if city_df is not None and not city_df.empty:
                            group_data.append(city_df)

                    except Exception as error:

                        city_name = city_info["name"]

                        print(
                            f"[NHÓM {group_number}] "
                            f"Không thu thập được {city_name}: "
                            f"{error}"
                        )

                        failed_in_group.append(
                            (city_id, city_info)
                        )

                        all_failed_cities.append(
                            city_name
                        )

        # ----------------------------------------------------
        # GỘP VÀ LƯU CHECKPOINT
        # ----------------------------------------------------

        if group_data:

            group_df = pd.concat(
                group_data,
                ignore_index=True
            )

            group_df["time"] = pd.to_datetime(
                group_df["time"]
            )

            group_df = (
                group_df
                .drop_duplicates(
                    subset=["city", "time"]
                )
                .sort_values(
                    ["city", "time"]
                )
                .reset_index(drop=True)
            )

            save_checkpoint(
                group_df,
                checkpoint_run_id,
                group_number
            )

            print(
                f"[NHÓM {group_number}] "
                f"Đã lưu checkpoint: "
                f"{len(group_df)} dòng."
            )

            all_city_data.append(group_df)

        else:

            print(
                f"[NHÓM {group_number}] "
                f"Không có dữ liệu để lưu checkpoint."
            )

        print(
            f"[NHÓM {group_number}] "
            f"Hoàn thành. "
            f"Thất bại: {len(failed_in_group)} địa điểm."
        )

    # ========================================================
    # 8. KIỂM TRA KẾT QUẢ TOÀN BỘ 63 ĐỊA ĐIỂM
    # ========================================================

    print("\n" + "=" * 60)
    print("TỔNG KẾT THU THẬP DỮ LIỆU")
    print("=" * 60)

    if not all_city_data:

        raise RuntimeError(
            "Không thu thập được dữ liệu từ bất kỳ địa điểm nào."
        )

    final_df = pd.concat(
        all_city_data,
        ignore_index=True
    )

    final_df["time"] = pd.to_datetime(
        final_df["time"]
    )

    final_df = final_df[
        final_df["time"] <= cutoff_time
    ].copy()

    final_df = (
        final_df
        .drop_duplicates(
            subset=["city", "time"]
        )
        .sort_values(
            ["city", "time"]
        )
        .reset_index(drop=True)
    )

    successful_cities = set(
        final_df["city"].unique()
    )

    expected_cities = {
        info["name"]
        for info in CITIES_63.values()
    }

    missing_cities = expected_cities - successful_cities

    print(
        f"Số địa điểm thu thập thành công: "
        f"{len(successful_cities)}/63"
    )

    print(
        f"Tổng số dòng dữ liệu: {len(final_df):,}"
    )

    if missing_cities:

        print("\nCác địa điểm chưa thu thập được:")

        for city in sorted(missing_cities):
            print(f"- {city}")

        raise RuntimeError(
            f"Thu thập dữ liệu chưa hoàn tất: "
            f"{len(missing_cities)} địa điểm bị thiếu. "
            f"Không ghi đè dữ liệu đầu ra."
        )

    # Kiểm tra dữ liệu không bị thiếu cột
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in final_df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"Dữ liệu cuối cùng thiếu cột: {missing_columns}"
        )

    # ========================================================
    # 9. LƯU FILE DỮ LIỆU CUỐI CÙNG
    # ========================================================

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = DATA_DIR / "latest_air_quality.csv"

    final_df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("THU THẬP DỮ LIỆU THÀNH CÔNG!")
    print("=" * 60)

    print(f"Số địa điểm: {len(successful_cities)}/63")
    print(f"Tổng số dòng: {len(final_df):,}")
    print(f"File đã lưu: {output_file}")
    print(f"Checkpoint Run ID: {checkpoint_run_id}")


# ============================================================
# 10. CHẠY CHƯƠNG TRÌNH
# ============================================================

if __name__ == "__main__":

    try:

        fetch_all_cities()

    except Exception as error:

        print("\n" + "=" * 60)
        print("PIPELINE THU THẬP DỮ LIỆU THẤT BẠI")
        print("=" * 60)

        print(f"Lỗi: {error}")

        raise
