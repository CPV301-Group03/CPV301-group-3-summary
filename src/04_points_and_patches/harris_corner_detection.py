# harris_corner_detection.py

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
    # Ảnh đường phố có nhiều góc cạnh, rất phù hợp cho Harris detector
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    
    print("Đang tải ảnh và phát hiện góc...")
    image_color = load_image_from_url(IMAGE_URL)

    if image_color is not None:
        # Chuyển ảnh sang ảnh xám để xử lý
        gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # 1. Áp dụng Harris Corner Detector
        # cv2.cornerHarris(image, blockSize, ksize, k)
        # - blockSize: Kích thước cửa sổ lân cận.
        # - ksize: Kích thước kernel của Sobel.
        # - k: Tham số tự do của Harris (thường từ 0.04 đến 0.06).
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

        # 2. Giãn nở kết quả để đánh dấu các góc rõ hơn
        dst = cv2.dilate(dst, None)

        # 3. Đặt ngưỡng để xác định các góc thực sự
        # Chỉ những điểm có giá trị 'R' lớn hơn 1% giá trị R tối đa mới được coi là góc.
        image_with_corners = image_color.copy()
        image_with_corners[dst > 0.01 * dst.max()] = [0, 0, 255] # Tô màu đỏ cho các góc

        # 4. Hiển thị kết quả
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image_with_corners, cv2.COLOR_BGR2RGB))
        plt.title("Harris Corner Detection")
        plt.axis('off')
        plt.show()