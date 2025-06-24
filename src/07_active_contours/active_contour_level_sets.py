# active_contour_level_sets.py

import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.segmentation import morphological_chan_vese
import requests
import cv2

def load_image_from_url(url: str, grayscale=True):
    """Tải ảnh từ URL và tùy chọn chuyển thành ảnh xám."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imdecode(image_array, mode)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

if __name__ == "__main__":
    # Ảnh có độ tương phản cao giữa nền và đối tượng hoạt động tốt nhất.
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg" # Ảnh chữ "morphology"

    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    image_gray = load_image_from_url(IMAGE_URL, grayscale=True)

    if image_gray is not None:
        # Chuyển ảnh sang dạng float trong khoảng [0, 1] là yêu cầu của hàm
        image = img_as_float(image_gray)

        print("Đang xử lý bằng phương pháp Level Sets (Chan-Vese)...")
        # 1. Áp dụng thuật toán Morphological Chan-Vese
        # *** SỬA LỖI: Đổi 'iterations' thành 'num_iter' ***
        segmentation_mask = morphological_chan_vese(image, num_iter=50, smoothing=1)
        
        # 2. Hiển thị kết quả
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        ax = axes.flatten()

        ax[0].imshow(image, cmap="gray")
        ax[0].set_title("1. Ảnh Gốc")
        ax[0].axis('off')

        ax[1].imshow(segmentation_mask, cmap="gray")
        ax[1].set_title("2. Mặt nạ Phân vùng")
        ax[1].axis('off')
        
        ax[2].imshow(image, cmap="gray")
        # Vẽ đường viền của mặt nạ lên ảnh gốc
        ax[2].contour(segmentation_mask, [0.5], colors='r')
        ax[2].set_title("3. Ảnh Gốc với Đường viền")
        ax[2].axis('off')

        fig.suptitle("Phân vùng bằng Level Sets (Morphological Chan-Vese)", fontsize=16)
        plt.tight_layout()
        plt.show()