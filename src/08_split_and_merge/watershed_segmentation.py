# watershed_segmentation.py

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
    IMAGE_URL = "https://as1.ftcdn.net/v2/jpg/01/16/57/34/1000_F_116573479_zks7Zu58thbbsLjl1MpjFOy3431LZiQO.jpg"
    
    print("Đang tải ảnh và xử lý bằng Watershed...")
    image_color = load_image_from_url(IMAGE_URL)

    if image_color is not None:
        # --- BƯỚC 1: Tiền xử lý ---
        gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)
        # Áp dụng ngưỡng Otsu để tách nền và vật thể
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # --- BƯỚC 2: Loại bỏ nhiễu và tìm vùng nền/vật thể chắc chắn ---
        # Loại bỏ nhiễu nhỏ bằng phép mở (opening)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Tìm vùng nền chắc chắn bằng phép giãn nở (dilation)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Tìm vùng vật thể chắc chắn bằng Distance Transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # Tìm vùng không xác định (biên giới giữa các đồng xu)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # --- BƯỚC 3: Tạo các "marker" cho Watershed ---
        _, markers = cv2.connectedComponents(sure_fg)
        # Thêm 1 vào tất cả các label để vùng nền chắc chắn là 1 thay vì 0
        markers = markers + 1
        # Đánh dấu vùng không xác định là 0
        markers[unknown == 255] = 0

        # --- BƯỚC 4: Áp dụng thuật toán Watershed ---
        markers = cv2.watershed(image_color, markers)
        image_with_boundaries = image_color.copy()
        # Các đường biên giới sẽ được đánh dấu bằng -1
        image_with_boundaries[markers == -1] = [255, 0, 0] # Tô màu đỏ cho đường biên

        # --- Hiển thị kết quả ---
        plt.figure(figsize=(12, 8))
        plt.suptitle("Phân vùng bằng Thuật toán Watershed", fontsize=16)
        
        plt.subplot(2, 3, 1), plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
        plt.title('1. Ảnh Gốc'), plt.axis('off')

        plt.subplot(2, 3, 2), plt.imshow(thresh, cmap='gray')
        plt.title('2. Ảnh sau Ngưỡng Otsu'), plt.axis('off')

        plt.subplot(2, 3, 3), plt.imshow(sure_bg, cmap='gray')
        plt.title('3. Vùng Nền Chắc chắn'), plt.axis('off')
        
        plt.subplot(2, 3, 4), plt.imshow(dist_transform, cmap='gray')
        plt.title('4. Distance Transform'), plt.axis('off')
        
        plt.subplot(2, 3, 5), plt.imshow(markers, cmap='jet')
        plt.title('5. Các Markers'), plt.axis('off')
        
        plt.subplot(2, 3, 6), plt.imshow(cv2.cvtColor(image_with_boundaries, cv2.COLOR_BGR2RGB))
        plt.title('6. Kết quả Phân vùng'), plt.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()