
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from huggingface_hub import HfApi, hf_hub_download

from config import (
    HF_TOKEN,
    DATASET_REPO_ID,
    MODEL_REPO_ID,
    MODEL_FILENAME,
    DATA_DIR,
    FEATURE_COLUMNS
)


# ============================================================
# 1. CẤU HÌNH
# ============================================================

INPUT_FILE = DATA_DIR / "prediction_features.csv"

OUTPUT_FILE = DATA_DIR / "latest_predictions.csv"

REMOTE_OUTPUT_FILE = "predictions/latest_predictions.csv"

RISK_LABELS = {
    0: "Thấp",
    1: "Trung bình",
    2: "Cao",
    3: "Rất cao"
}


# ============================================================
# 2. TẢI MÔ HÌNH CATBOOST
# ============================================================

def load_model():

    print("Đang tải mô hình từ Hugging Face...")

    # Kiểm tra cấu hình, tuyệt đối không in HF_TOKEN
    print("\nKIỂM TRA CẤU HÌNH:")
    print("MODEL_REPO_ID:", MODEL_REPO_ID)
    print("MODEL_FILENAME:", MODEL_FILENAME)
    print("HF_TOKEN tồn tại:", bool(HF_TOKEN))

    if not MODEL_REPO_ID:
        raise ValueError("MODEL_REPO_ID đang bị trống!")

    if not MODEL_FILENAME:
        raise ValueError("MODEL_FILENAME đang bị trống!")

    if not HF_TOKEN:
        raise ValueError("HF_TOKEN chưa được cấu hình!")

    print("\nĐang kết nối Model Repository...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILENAME,
        repo_type="model",
        token=HF_TOKEN
    )

    print("Tải file mô hình thành công!")
    print("Đường dẫn file mô hình:", model_path)

    model = CatBoostClassifier()

    model.load_model(model_path)

    print("Đã load mô hình CatBoost!")

    # Kiểm tra số lượng features
    if len(model.feature_names_) != 33:
        raise ValueError(
            f"Mô hình yêu cầu {len(model.feature_names_)} features, "
            "không phải 33!"
        )

    # Kiểm tra thứ tự features
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise ValueError(
            "Thứ tự features không khớp với mô hình CatBoost!"
        )

    # Kiểm tra các lớp dự báo
    if list(model.classes_) != [0, 1, 2, 3]:
        raise ValueError(
            f"Các lớp của mô hình không hợp lệ: {model.classes_}"
        )

    print("\nKiểm tra mô hình thành công!")
    print("Số features:", len(model.feature_names_))
    print("Các lớp:", model.classes_)

    return model


# ============================================================
# 3. ĐỌC DỮ LIỆU ĐẦU VÀO
# ============================================================

def load_prediction_data():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "time",
        "prediction_time"
    ] + FEATURE_COLUMNS

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Dữ liệu thiếu các cột: {missing}"
        )

    if len(df) != 63:
        raise ValueError(
            f"Cần 63 dòng dự báo nhưng nhận được {len(df)} dòng!"
        )

    if df["city"].nunique() != 63:
        raise ValueError(
            "Dữ liệu không có đủ 63 địa điểm riêng biệt!"
        )

    df["city"] = df["city"].astype(str)
    df["region"] = df["region"].astype(str)

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError(
            "Dữ liệu đầu vào còn giá trị thiếu!"
        )

    print("\nĐọc dữ liệu thành công!")
    print("Số dòng:", len(df))
    print("Số địa điểm:", df["city"].nunique())

    return df


# ============================================================
# 4. THỰC HIỆN DỰ BÁO
# ============================================================

def predict_air_quality(model, df):

    print("\nĐang thực hiện dự báo...")

    X = df[FEATURE_COLUMNS].copy()

    # Dự báo nhãn
    predictions = model.predict(X)

    predictions = np.asarray(
        predictions
    ).reshape(-1).astype(int)

    # Xác suất của từng lớp
    probabilities = model.predict_proba(X)

    # Tạo bảng kết quả
    result = df[
        ["city", "region", "time", "prediction_time"]
    ].copy()

    result["predicted_risk_level"] = predictions

    result["predicted_risk_name"] = [
        RISK_LABELS[value]
        for value in predictions
    ]

    # Thêm xác suất dự báo
    for index, risk_level in enumerate(model.classes_):

        result[
            f"probability_level_{risk_level}"
        ] = probabilities[:, index]

    print("Dự báo thành công!")

    return result


# ============================================================
# 5. LƯU KẾT QUẢ DỰ BÁO
# ============================================================

def save_predictions(result):

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nĐã lưu kết quả tại:")
    print(OUTPUT_FILE)


# ============================================================
# 6. UPLOAD KẾT QUẢ LÊN HUGGING FACE
# ============================================================

def upload_predictions():

    print("\nĐang upload kết quả dự báo...")

    api = HfApi(token=HF_TOKEN)

    api.upload_file(
        path_or_fileobj=str(OUTPUT_FILE),
        path_in_repo=REMOTE_OUTPUT_FILE,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Update air quality predictions"
    )

    print("Upload kết quả dự báo thành công!")
    print("File:", REMOTE_OUTPUT_FILE)


# ============================================================
# 7. CHƯƠNG TRÌNH CHÍNH
# ============================================================

def main():

    print("=" * 60)
    print("VIETNAM AIR QUALITY PREDICTION")
    print("=" * 60)

    # Tải mô hình
    model = load_model()

    # Đọc features
    df = load_prediction_data()

    # Dự báo
    result = predict_air_quality(
        model,
        df
    )

    # Lưu kết quả
    save_predictions(result)

    # Upload kết quả
    upload_predictions()

    # Hiển thị kết quả
    print("\n" + "=" * 60)
    print("KẾT QUẢ DỰ BÁO 63 ĐỊA ĐIỂM")
    print("=" * 60)

    print(
        result[
            [
                "city",
                "region",
                "prediction_time",
                "predicted_risk_level",
                "predicted_risk_name"
            ]
        ].to_string(index=False)
    )

    print("\nHOÀN THÀNH DỰ BÁO!")


if __name__ == "__main__":
    main()
