# kmeans_color_segmentation.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL và chuyển sang định dạng RGB."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None: raise IOError("Không thể giải mã ảnh.")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    # Số lượng cụm màu mong muốn (K)
    K = 5 
    
    print(f"Đang tải ảnh và phân vùng với K={K} cụm màu...")
    image = load_image_from_url(IMAGE_URL)

    if image is not None:
        # 1. Chuẩn bị dữ liệu
        # Chuyển đổi ảnh từ 2D (height x width x 3) thành một danh sách các pixel (N x 3)
        pixel_vals = image.reshape((-1, 3))
        # Chuyển sang kiểu float32 là yêu cầu của K-Means trong OpenCV
        pixel_vals = np.float32(pixel_vals)

        # 2. Áp dụng K-Means
        # Định nghĩa tiêu chí dừng: lặp tối đa 100 lần hoặc khi độ chính xác đạt 0.2
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        # Thực hiện phân cụm
        _, labels, centers = cv2.kmeans(pixel_vals, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 3. Tái tạo ảnh đã phân vùng
        # Chuyển đổi các tâm cụm (centers) về lại kiểu uint8
        centers = np.uint8(centers)
        # Ánh xạ mỗi pixel về màu của tâm cụm tương ứng
        segmented_data = centers[labels.flatten()]
        # Chuyển đổi lại dữ liệu thành ảnh 2D
        segmented_image = segmented_data.reshape((image.shape))

        # 4. Hiển thị kết quả
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"Phân vùng ảnh bằng K-Means (K={K})", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Ảnh Gốc")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title("Ảnh đã Phân vùng")
        plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()