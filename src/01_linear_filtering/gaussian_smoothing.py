# gaussian_smoothing.py

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

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 9, sigma_x: float = 0):
    """
    Áp dụng bộ lọc làm mờ Gauss (Gaussian Blur) lên một hình ảnh.

    Bộ lọc Gauss làm mờ ảnh bằng cách sử dụng một kernel có trọng số tuân theo phân phối Gauss.
    Các pixel ở gần trung tâm có trọng số cao hơn.

    Args:
        image (np.ndarray): Hình ảnh đầu vào (định dạng BGR của OpenCV).
        kernel_size (int): Kích thước của kernel (phải là số lẻ, ví dụ: 3, 5, 9).
        sigma_x (float): Độ lệch chuẩn theo trục X. Nếu để là 0, nó sẽ được
                         tính tự động từ kernel_size.

    Returns:
        numpy.ndarray: Hình ảnh đã được làm mờ.
    """
    if kernel_size % 2 == 0 or kernel_size <= 0:
        raise ValueError("Kích thước kernel (kernel_size) phải là một số lẻ dương.")
        
    # Sử dụng hàm cv2.GaussianBlur của OpenCV.
    # ksize là một tuple (width, height).
    # sigmaX điều khiển mức độ mờ, sigmaY (nếu không được chỉ định) sẽ bằng sigmaX.
    blurred_image = cv2.GaussianBlur(src=image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x)
    
    return blurred_image

def display_images(original_image: np.ndarray, filtered_image: np.ndarray, kernel_size: int, sigma_x: float):
    """
    Hiển thị hình ảnh gốc và hình ảnh đã lọc cạnh nhau để so sánh.
    """
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 6))
    plt.suptitle("Ứng dụng Làm mờ Gauss (Gaussian Smoothing)", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Ảnh Gốc")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_rgb)
    plt.title(f"Ảnh đã lọc (Kernel: {kernel_size}x{kernel_size}, SigmaX: {sigma_x})")
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    
    # Kích thước kernel (phải là số lẻ)
    KERNEL_SIZE = 15
    # Độ lệch chuẩn. 0 có nghĩa là OpenCV sẽ tự tính toán.
    # Giá trị lớn hơn sẽ làm ảnh mờ hơn.
    SIGMA_X = 0

    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    original_image = load_image_from_url(IMAGE_URL)

    if original_image is not None:
        print(f"Áp dụng bộ lọc Gauss với kernel={KERNEL_SIZE}, sigmaX={SIGMA_X}...")
        
        try:
            blurred_image = apply_gaussian_blur(original_image, KERNEL_SIZE, SIGMA_X)
            print("Hiển thị kết quả...")
            display_images(original_image, blurred_image, KERNEL_SIZE, SIGMA_X)
        except ValueError as e:
            print(f"Lỗi: {e}")