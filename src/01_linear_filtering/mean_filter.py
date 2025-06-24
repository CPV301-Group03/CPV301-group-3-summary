# mean_filter.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """
    Tải một hình ảnh từ một URL và chuyển đổi nó thành định dạng OpenCV.

    Args:
        url (str): Địa chỉ URL của hình ảnh.

    Returns:
        numpy.ndarray: Hình ảnh dưới dạng một mảng NumPy (định dạng BGR của OpenCV),
                       hoặc None nếu không thể tải ảnh.
    """
    try:
        # Gửi yêu cầu GET đến URL để lấy dữ liệu ảnh
        response = requests.get(url, timeout=10)
        # Kiểm tra nếu yêu cầu thành công (status code 200)
        response.raise_for_status()

        # Chuyển đổi nội dung nhận được thành một mảng byte
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        
        # Giải mã mảng byte thành một hình ảnh màu OpenCV (BGR)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Lỗi: Không thể giải mã hình ảnh từ URL.")
            return None
            
        return image

    except requests.exceptions.RequestException as e:
        print(f"Lỗi: Không thể tải hình ảnh từ URL. {e}")
        return None

def apply_mean_filter(image: np.ndarray, kernel_size: int = 5):
    """
    Áp dụng bộ lọc trung bình (mean filter) lên một hình ảnh.

    Bộ lọc trung bình làm mờ ảnh bằng cách thay thế giá trị của mỗi pixel
    bằng giá trị trung bình của các pixel lân cận trong một cửa sổ (kernel).

    Args:
        image (np.ndarray): Hình ảnh đầu vào (định dạng BGR của OpenCV).
        kernel_size (int): Kích thước của kernel (phải là số lẻ, ví dụ: 3, 5, 7).

    Returns:
        numpy.ndarray: Hình ảnh đã được làm mờ.
    """
    # Kiểm tra để đảm bảo kích thước kernel là số lẻ
    if kernel_size % 2 == 0:
        raise ValueError("Kích thước kernel (kernel_size) phải là một số lẻ.")
        
    # Sử dụng hàm cv2.blur của OpenCV để áp dụng bộ lọc trung bình.
    # Hàm này rất hiệu quả và được tối ưu hóa.
    # (kernel_size, kernel_size) là kích thước của cửa sổ trượt.
    blurred_image = cv2.blur(src=image, ksize=(kernel_size, kernel_size))
    
    return blurred_image

def display_images(original_image: np.ndarray, filtered_image: np.ndarray, kernel_size: int):
    """
    Hiển thị hình ảnh gốc và hình ảnh đã lọc cạnh nhau để so sánh.

    Args:
        original_image (np.ndarray): Hình ảnh gốc.
        filtered_image (np.ndarray): Hình ảnh đã qua bộ lọc.
        kernel_size (int): Kích thước kernel được sử dụng.
    """
    # Chuyển đổi không gian màu từ BGR (OpenCV) sang RGB (Matplotlib)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    filtered_rgb = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)

    # Tạo một figure để chứa các biểu đồ
    plt.figure(figsize=(12, 6))
    plt.suptitle("Ứng dụng Bộ lọc Trung bình (Mean Filter)", fontsize=16)

    # Biểu đồ cho ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Ảnh Gốc")
    plt.axis('off')  # Ẩn các trục tọa độ

    # Biểu đồ cho ảnh đã lọc
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_rgb)
    plt.title(f"Ảnh đã lọc (Kernel: {kernel_size}x{kernel_size})")
    plt.axis('off')

    # Hiển thị cửa sổ biểu đồ
    plt.show()

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    # --- THAM SỐ CẤU HÌNH ---
    # Bạn có thể thay đổi URL này bằng bất kỳ URL ảnh công khai nào
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    KERNEL_SIZE = 9  # Kích thước kernel phải là số lẻ (ví dụ: 3, 5, 9, 15)

    print(f"Đang tải ảnh từ: {IMAGE_URL}...")
    original_image = load_image_from_url(IMAGE_URL)

    # Chỉ tiếp tục nếu ảnh được tải thành công
    if original_image is not None:
        print(f"Áp dụng bộ lọc trung bình với kernel size = {KERNEL_SIZE}x{KERNEL_SIZE}...")
        
        try:
            # Áp dụng bộ lọc
            blurred_image = apply_mean_filter(original_image, KERNEL_SIZE)
            
            print("Hiển thị kết quả...")
            # Hiển thị ảnh gốc và ảnh đã xử lý
            display_images(original_image, blurred_image, KERNEL_SIZE)
            
        except ValueError as e:
            print(f"Lỗi: {e}")