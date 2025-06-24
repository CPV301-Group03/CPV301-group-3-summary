# convolutional_line_detection.py

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

def apply_line_kernel(image, kernel):
    """Áp dụng một kernel tích chập lên ảnh."""
    # Áp dụng tích chập và lấy giá trị tuyệt đối để phát hiện cả đường sáng và tối
    filtered = cv2.filter2D(image, -1, kernel)
    return np.uint8(np.absolute(filtered))


if __name__ == "__main__":
    # Ảnh có nhiều đường thẳng theo các hướng khác nhau
    IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
    
    print("Đang tải ảnh và xử lý...")
    image = load_image_from_url(IMAGE_URL)
    
    if image is not None:
        # --- Định nghĩa các Kernel từ slide 11 ---
        # Các kernel này được thiết kế để phát hiện đường sáng trên nền tối.
        kernel_horizontal = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])
        kernel_vertical = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
        kernel_oblique_45 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])
        kernel_oblique_135 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

        # --- Áp dụng từng kernel ---
        horizontal_lines = apply_line_kernel(image, kernel_horizontal)
        vertical_lines = apply_line_kernel(image, kernel_vertical)
        oblique_45_lines = apply_line_kernel(image, kernel_oblique_45)
        oblique_135_lines = apply_line_kernel(image, kernel_oblique_135)

        # --- Hiển thị kết quả ---
        plt.figure(figsize=(12, 10))
        plt.suptitle("Phát hiện đường thẳng bằng Tích chập (Convolution)", fontsize=16)

        plt.subplot(3, 2, 1), plt.imshow(image, cmap='gray'), plt.title('Ảnh Gốc')
        
        plt.subplot(3, 2, 3), plt.imshow(horizontal_lines, cmap='gray')
        plt.title('Phát hiện đường Ngang')
        
        plt.subplot(3, 2, 4), plt.imshow(vertical_lines, cmap='gray')
        plt.title('Phát hiện đường Dọc')
        
        plt.subplot(3, 2, 5), plt.imshow(oblique_45_lines, cmap='gray')
        plt.title('Phát hiện đường Chéo (+45°)')
        
        plt.subplot(3, 2, 6), plt.imshow(oblique_135_lines, cmap='gray')
        plt.title('Phát hiện đường Chéo (-45°)')

        # Xóa các trục thừa
        plt.subplot(3, 2, 2).axis('off')
        
        for i in range(1, 7):
            plt.subplot(3, 2, i).axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()