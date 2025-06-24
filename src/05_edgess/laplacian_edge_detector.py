# laplacian_edge_detector.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL và chuyển thành ảnh xám."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"

    print("Đang tải ảnh và xử lý...")
    image = load_image_from_url(IMAGE_URL)

    if image is not None:
        # 1. Làm mờ ảnh bằng Gaussian Filter để giảm nhiễu (Bước 'G' trong LoG)
        blurred_image = cv2.GaussianBlur(image, (3, 3), 0)

        # 2. Áp dụng toán tử Laplacian
        # Sử dụng cv2.CV_64F để giữ lại các giá trị âm (quan trọng cho zero-crossing)
        laplacian = cv2.Laplacian(blurred_image, cv2.CV_64F)
        
        # Chuyển đổi về uint8 để hiển thị
        laplacian_display = cv2.convertScaleAbs(laplacian)

        # 3. Hiển thị kết quả
        plt.figure(figsize=(18, 6))
        plt.suptitle("Phát hiện cạnh bằng Laplacian of Gaussian (LoG)", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Ảnh Gốc')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(blurred_image, cmap='gray')
        plt.title('Ảnh sau khi làm mờ Gaussian')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(laplacian_display, cmap='gray')
        plt.title('Cạnh phát hiện bằng Laplacian')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()