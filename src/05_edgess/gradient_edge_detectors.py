# gradient_edge_detectors.py

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
    # Sử dụng ảnh có nhiều đường thẳng ngang và dọc để dễ so sánh
    IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"

    print("Đang tải ảnh và xử lý...")
    image = load_image_from_url(IMAGE_URL)

    if image is not None:
        # --- 1. Sobel Operator (Có sẵn trong OpenCV) ---
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_x = np.uint8(np.absolute(sobel_x))
        sobel_y = np.uint8(np.absolute(sobel_y))
        sobel_combined = cv2.bitwise_or(sobel_x, sobel_y)

        # --- 2. Prewitt Operator (Tự định nghĩa kernel) ---
        kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(image, -1, kernel_prewitt_x)
        prewitt_y = cv2.filter2D(image, -1, kernel_prewitt_y)
        prewitt_combined = cv2.bitwise_or(prewitt_x, prewitt_y)

        # --- 3. Robert's Cross Operator (Tự định nghĩa kernel) ---
        kernel_robert_x = np.array([[1, 0], [0, -1]])
        kernel_robert_y = np.array([[0, 1], [-1, 0]])
        robert_x = cv2.filter2D(image, -1, kernel_robert_x)
        robert_y = cv2.filter2D(image, -1, kernel_robert_y)
        robert_combined = cv2.bitwise_or(robert_x, robert_y)

        # --- Hiển thị kết quả ---
        plt.figure(figsize=(16, 8))
        plt.suptitle("So sánh các Toán tử phát hiện cạnh dựa trên Gradient", fontsize=16)

        plt.subplot(2, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh Gốc')
        plt.subplot(2, 4, 2), plt.imshow(sobel_combined, cmap='gray'), plt.title('Sobel')
        plt.subplot(2, 4, 3), plt.imshow(prewitt_combined, cmap='gray'), plt.title('Prewitt')
        plt.subplot(2, 4, 4), plt.imshow(robert_combined, cmap='gray'), plt.title("Robert's Cross")
        
        # Hiển thị riêng các thành phần x và y của Sobel
        plt.subplot(2, 4, 5), plt.axis('off') # Trống
        plt.subplot(2, 4, 6), plt.imshow(sobel_x, cmap='gray'), plt.title('Sobel X (Cạnh dọc)')
        plt.subplot(2, 4, 7), plt.imshow(sobel_y, cmap='gray'), plt.title('Sobel Y (Cạnh ngang)')
        plt.subplot(2, 4, 8), plt.axis('off') # Trống
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()