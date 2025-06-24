# bilateral_filter.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """
    Tải một hình ảnh từ một URL và chuyển đổi nó thành định dạng OpenCV.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            print("Lỗi: Không thể giải mã hình ảnh từ URL.")
            return None
        return image
    except requests.exceptions.RequestException as e:
        print(f"Lỗi: Không thể tải hình ảnh từ URL. {e}")
        return None

def apply_bilateral_filter(image: np.ndarray, d: int, sigma_color: float, sigma_space: float):
    """
    Áp dụng bộ lọc song phương (Bilateral Filter) lên một hình ảnh.

    Bộ lọc này rất hiệu quả trong việc giảm nhiễu mà vẫn giữ được các cạnh sắc nét.
    Nó xem xét cả sự khác biệt về không gian (khoảng cách) và sự khác biệt về màu sắc.

    Args:
        image (np.ndarray): Hình ảnh đầu vào (định dạng BGR của OpenCV).
        d (int): Đường kính của vùng lân cận của mỗi pixel.
        sigma_color (float): Độ lệch chuẩn trong không gian màu. Giá trị lớn hơn có nghĩa là
                             các màu xa nhau hơn sẽ được tính vào vùng lân cận để làm mờ.
        sigma_space (float): Độ lệch chuẩn trong không gian tọa độ. Giá trị lớn hơn có nghĩa là
                             các pixel ở xa hơn sẽ ảnh hưởng đến phép tính.

    Returns:
        numpy.ndarray: Hình ảnh đã được lọc.
    """
    # Sử dụng hàm cv2.bilateralFilter của OpenCV.
    filtered_image = cv2.bilateralFilter(src=image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    
    return filtered_image

def display_images(original_image: np.ndarray, filtered_image: np.ndarray, d: int, sc: float, ss: float):
    """
    Hiển thị hình ảnh gốc và hình ảnh đã lọc cạnh nhau để so sánh.
    """
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Ứng dụng Bộ lọc Song phương (Bilateral Filter)", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Ảnh Gốc")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_rgb)
    plt.title(f"Ảnh đã lọc (d={d}, σ_color={sc}, σ_space={ss})")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    
    # --- THAM SỐ CỦA BILATERAL FILTER ---
    # Đường kính vùng lân cận. Giá trị lớn hơn cho hiệu ứng mạnh hơn nhưng chậm hơn.
    D_PARAM = 15
    # Độ lệch chuẩn màu. Giá trị lớn hơn làm các màu khác biệt hơn bị trộn lẫn.
    SIGMA_COLOR = 80
    # Độ lệch chuẩn không gian. Giá trị lớn hơn làm các pixel ở xa hơn ảnh hưởng đến nhau.
    SIGMA_SPACE = 80

    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    original_image = load_image_from_url(IMAGE_URL)

    if original_image is not None:
        print(f"Áp dụng bộ lọc song phương với d={D_PARAM}, sigmaColor={SIGMA_COLOR}, sigmaSpace={SIGMA_SPACE}...")
        
        filtered_image = apply_bilateral_filter(original_image, D_PARAM, SIGMA_COLOR, SIGMA_SPACE)
        
        print("Hiển thị kết quả...")
        display_images(original_image, filtered_image, D_PARAM, SIGMA_COLOR, SIGMA_SPACE)