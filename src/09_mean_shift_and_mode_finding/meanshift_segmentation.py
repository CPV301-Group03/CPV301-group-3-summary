# meanshift_segmentation.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None: raise IOError("Không thể giải mã ảnh.")
        return image_bgr
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    
    print("Đang tải ảnh và xử lý bằng Mean Shift...")
    image_bgr = load_image_from_url(IMAGE_URL)

    if image_bgr is not None:
        # 1. Áp dụng Mean Shift
        # cv2.pyrMeanShiftFiltering(src, sp, sr)
        # - sp (spatial window radius): Bán kính không gian. Các pixel trong bán kính này sẽ được
        #   xem xét trong cửa sổ dịch chuyển.
        # - sr (color window radius): Bán kính màu. Các pixel có màu trong khoảng này
        #   so với pixel trung tâm sẽ được tính vào trung bình.
        # Các giá trị này ảnh hưởng lớn đến kết quả. Giá trị nhỏ sẽ giữ lại nhiều chi tiết hơn,
        # giá trị lớn sẽ tạo ra các vùng màu lớn và mượt hơn.
        segmented_image_bgr = cv2.pyrMeanShiftFiltering(image_bgr, sp=21, sr=51)

        # 2. Hiển thị kết quả
        # Chuyển đổi sang RGB để Matplotlib hiển thị đúng
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        segmented_image_rgb = cv2.cvtColor(segmented_image_bgr, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 6))
        plt.suptitle("Phân vùng ảnh bằng Mean Shift", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Ảnh Gốc")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image_rgb)
        plt.title("Ảnh đã Phân vùng")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()