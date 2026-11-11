import os

# 1. HUGGING FACE REPOSITORIES

HF_TOKEN = os.environ["HF_TOKEN"]

DATASET_REPO = os.environ["HF_DATASET_REPO"]
MODEL_REPO = os.environ["HF_MODEL_REPO"]

# 2. FILE NAMES

DATASET_FILENAME = "vietnam_air_quality_multifactor_full.csv"
MODEL_FILENAME = "catboost_air_quality.cbm"

PREDICTION_FILENAME = "vietnam_air_quality_predictions.csv"


# 3. SYSTEM CONFIGURATION

TIMEZONE = "Asia/Ho_Chi_Minh"

# Dự báo chất lượng không khí sau 3 giờ
PREDICTION_HORIZON_HOURS = 3

# Số tỉnh/thành theo phạm vi dữ liệu lịch sử
TOTAL_CITIES = 63


# 4. MODEL FEATURES

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
