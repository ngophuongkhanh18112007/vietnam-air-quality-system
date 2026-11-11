# KIỂM TRA DANH SÁCH 63 TỈNH/THÀNH PHỐ

assert len(CITIES_63) == 63, (
    f"Số địa điểm không đúng: {len(CITIES_63)}"
)

# Kiểm tra phân bố theo vùng
region_counts = {}

for city_id, info in CITIES_63.items():
    region = info["region"]
    region_counts[region] = region_counts.get(region, 0) + 1

assert region_counts == {
    "Mien_Bac": 25,
    "Mien_Trung": 19,
    "Mien_Nam": 19
}, f"Phân bố vùng không đúng: {region_counts}"

# Kiểm tra thông tin từng địa điểm
for city_id, info in CITIES_63.items():
    assert all(
        key in info for key in ["name", "region", "lat", "lon"]
    ), f"Thiếu thông tin tại: {city_id}"

print("Đã kiểm tra danh sách 63 địa điểm thành công!")
print("Phân bố:", region_counts)
