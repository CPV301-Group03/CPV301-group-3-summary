# graph_based_segmentation.py

import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.util import img_as_float
import requests
import cv2
import numpy as np

def load_image_from_url(url: str):
    """Tải một hình ảnh từ URL và chuyển đổi nó thành định dạng RGB."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image_bgr is None: raise IOError("Không thể giải mã ảnh.")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return image_rgb
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

if __name__ == "__main__":
    IMAGE_URL = "https://as1.ftcdn.net/v2/jpg/01/16/57/34/1000_F_116573479_zks7Zu58thbbsLjl1MpjFOy3431LZiQO.jpg"

    print("Đang tải ảnh và xử lý bằng Graph-Based Segmentation...")
    image_color = load_image_from_url(IMAGE_URL)

    if image_color is not None:
        # Chuyển ảnh sang dạng float là yêu cầu của hàm
        image_float = img_as_float(image_color)
        
        # 1. Áp dụng thuật toán Felzenszwalb
        # - scale: Giá trị càng lớn, các vùng phân đoạn càng lớn.
        # - sigma: Độ mượt của ảnh trước khi phân vùng.
        # - min_size: Kích thước tối thiểu (pixel) của một vùng.
        segments_fz = felzenszwalb(image_float, scale=200, sigma=0.5, min_size=100)
        
        # 2. Hiển thị kết quả
        # mark_boundaries sẽ vẽ đường viền của các vùng lên ảnh gốc.
        segmented_image = mark_boundaries(image_float, segments_fz, color=(1, 0, 0)) # Viền đỏ

        print(f"Tìm thấy {len(np.unique(segments_fz))} vùng.")

        plt.figure(figsize=(18, 6))
        plt.suptitle("Phân vùng bằng Graph-Based (Felzenszwalb)", fontsize=16)

        plt.subplot(1, 2, 1)
        plt.imshow(image_color)
        plt.title("Ảnh Gốc")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(segmented_image)
        plt.title("Ảnh đã Phân vùng")
        plt.axis('off')

        plt.tight_layout()
        plt.show()