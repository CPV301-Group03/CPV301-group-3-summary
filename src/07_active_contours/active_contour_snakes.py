# active_contour_snakes.py (Phiên bản đã sửa)

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import requests
import cv2

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

def create_circular_contour(center_y, center_x, radius, num_points=100):
    """Tạo một đường viền hình tròn ban đầu."""
    s = np.linspace(0, 2 * np.pi, num_points)
    y = center_y + radius * np.sin(s)
    x = center_x + radius * np.cos(s)
    return np.array([x, y]).T

if __name__ == "__main__":
    IMAGE_URL = "https://th.bing.com/th/id/R.e3c0b57e39b9038b5a33a28d7953f24d?rik=BcGZS3yNHRXECA&pid=ImgRaw&r=0"
    
    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    image_color = load_image_from_url(IMAGE_URL)

    if image_color is not None:
        image_gray = rgb2gray(image_color)

        # *** THAY ĐỔI QUAN TRỌNG 1: Đặt lại đường viền ban đầu ***
        # Chúng ta sẽ tạo một vòng tròn lớn hơn bao quanh toàn bộ khuôn mặt
        # để nó cắt qua các cạnh mạnh như cằm, má, và tóc.
        h, w = image_gray.shape
        center_y, center_x, radius = h / 2, w / 2, 250
        initial_contour = create_circular_contour(center_y, center_x, radius, num_points=200)

        print("Đang xử lý bằng mô hình Snakes với tham số đã tinh chỉnh...")
        
        # *** THAY ĐỔI QUAN TRỌNG 2: Tinh chỉnh các tham số ***
        # - w_edge: Tăng trọng số cho "lực hút" của cạnh.
        # - beta: Giảm độ cứng để "con rắn" có thể uốn cong linh hoạt hơn.
        snake = active_contour(
            gaussian(image_gray, 5, preserve_range=False), # Tăng blur một chút
            initial_contour,
            alpha=0.01,
            beta=5.0,        # Giảm độ cứng
            w_edge=5,        # Tăng lực hút của cạnh
            w_line=0,
            gamma=0.001
        )

        # Hiển thị kết quả
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(image_color)
        ax.plot(initial_contour[:, 0], initial_contour[:, 1], '--r', lw=2, label='Đường viền ban đầu')
        ax.plot(snake[:, 0], snake[:, 1], '-b', lw=2, label='Đường viền cuối cùng (Snake)')
        ax.set_title("Phân vùng bằng Active Contour (Snakes)", fontsize=16)
        ax.axis('off')
        ax.legend()
        plt.tight_layout()
        plt.show()