
import os
from pathlib import Path

# 1. HUGGING FACE CONFIGURATION

HF_TOKEN = os.environ["HF_TOKEN"]

DATASET_REPO_ID = os.environ["HF_DATASET_REPO"]
MODEL_REPO_ID = os.environ["HF_MODEL_REPO"]

# Giữ tên biến cũ để tương thích nếu file nào đang sử dụng
DATASET_REPO = DATASET_REPO_ID
MODEL_REPO = MODEL_REPO_ID

# 2. FILE CONFIGURATION

DATASET_FILENAME = "vietnam_air_quality_multifactor_full.csv"
MODEL_FILENAME = "catboost_air_quality.cbm"

PREDICTION_FILENAME = "latest_predictions.csv"

# 3. SYSTEM CONFIGURATION

TIMEZONE = "Asia/Ho_Chi_Minh"

PREDICTION_HORIZON_HOURS = 3

TOTAL_CITIES = 63

# 4. PROJECT DIRECTORIES

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# 5. MODEL FEATURES

FEATURE_COLUMNS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",

    "pm2_5_lag_1h", "pm2_5_lag_3h",
    "pm2_5_lag_6h", "pm2_5_lag_24h",

    "pm10_lag_1h", "pm10_lag_3h",
    "pm10_lag_6h", "pm10_lag_24h",

    "carbon_monoxide_lag_1h",
    "carbon_monoxide_lag_3h",
    "carbon_monoxide_lag_6h",
    "carbon_monoxide_lag_24h",

    "nitrogen_dioxide_lag_1h",
    "nitrogen_dioxide_lag_3h",
    "nitrogen_dioxide_lag_6h",
    "nitrogen_dioxide_lag_24h",

    "pm2_5_rolling_3h", "pm2_5_rolling_6h",
    "pm10_rolling_3h", "pm10_rolling_6h",

    "carbon_monoxide_rolling_3h",
    "carbon_monoxide_rolling_6h",

    "nitrogen_dioxide_rolling_3h",
    "nitrogen_dioxide_rolling_6h",

    "hour", "day_of_week", "month", "city", "region"
]

assert len(FEATURE_COLUMNS) == 33, (
    f"Expected 33 features, got {len(FEATURE_COLUMNS)}"
)
