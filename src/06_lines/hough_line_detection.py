# hough_line_detection.py

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
    # Ảnh có nhiều đường thẳng rõ ràng như Sudoku hoặc đường cao tốc là lý tưởng.
    IMAGE_URL = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/sudoku.png"
    
    print("Đang tải ảnh và xử lý...")
    image_color = load_image_from_url(IMAGE_URL)

    if image_color is not None:
        # --- BƯỚC 1 & 2: Tiền xử lý và Phát hiện cạnh ---
        # Chuyển sang ảnh xám
        gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        # Phát hiện cạnh bằng Canny làm đầu vào cho Hough Transform
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # --- BƯỚC 3: Áp dụng Hough Transform ---
        # cv2.HoughLinesP (Probabilistic Hough Line Transform) hiệu quả hơn
        # - rho: Độ phân giải khoảng cách (pixel)
        # - theta: Độ phân giải góc (radian)
        # - threshold: Ngưỡng tối thiểu số "phiếu bầu" để coi là một đường thẳng
        # - minLineLength: Chiều dài tối thiểu của một đoạn thẳng
        # - maxLineGap: Khoảng trống tối đa giữa các điểm trên cùng một đoạn thẳng
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, 
                                minLineLength=100, maxLineGap=10)

        # --- BƯỚC 4: Vẽ các đường thẳng đã tìm thấy lên ảnh gốc ---
        image_with_lines = image_color.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2) # Vẽ đường màu đỏ

        # --- Hiển thị quy trình ---
        plt.figure(figsize=(18, 6))
        plt.suptitle("Phát hiện đường thẳng bằng Hough Transform", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
        plt.title('1. Ảnh Gốc')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('2. Cạnh phát hiện (Canny)')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB))
        plt.title('3. Đường thẳng phát hiện')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()