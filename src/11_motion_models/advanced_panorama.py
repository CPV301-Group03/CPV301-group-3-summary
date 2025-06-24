# advanced_panorama.py

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
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    # Sử dụng một bộ ảnh có góc nhìn rộng để thấy rõ hiệu quả
    URLS = [
        "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat1.jpg",
        "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat2.jpg",
        "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat3.jpg",
        "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat4.jpg",
    ]

    print("Đang tải các ảnh...")
    images = [load_image_from_url(url) for url in URLS]
    # Loại bỏ các ảnh không tải được
    images = [img for img in images if img is not None]

    if len(images) > 1:
        print("Đang thực hiện ghép ảnh bằng cv2.Stitcher...")
        # 1. Khởi tạo Stitcher
        # OpenCV sẽ tự động chọn mô hình chiếu tốt nhất (thường là spherical hoặc cylindrical cho góc rộng)
        stitcher = cv2.Stitcher_create()
        
        # 2. Thực hiện ghép ảnh
        (status, stitched_image) = stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            print("Ghép ảnh thành công!")
            # 3. Hiển thị kết quả
            plt.figure(figsize=(20, 10))
            plt.imshow(cv2.cvtColor(stitched_image, cv2.COLOR_BGR2RGB))
            plt.title("Panorama tạo bằng cv2.Stitcher (Mô hình tự động)")
            plt.axis('off')
            plt.show()
        else:
            print(f"Ghép ảnh thất bại! Mã lỗi: {status}")