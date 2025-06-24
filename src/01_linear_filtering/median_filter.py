# median_filter.py

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
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Ném lỗi nếu yêu cầu HTTP không thành công

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Lỗi: Không thể giải mã hình ảnh từ URL.")
            return None
            
        return image

    except requests.exceptions.RequestException as e:
        print(f"Lỗi: Không thể tải hình ảnh từ URL. {e}")
        return None

def apply_median_filter(image: np.ndarray, kernel_size: int = 5):
    """
    Áp dụng bộ lọc trung vị (median filter) lên một hình ảnh.

    Bộ lọc trung vị rất hiệu quả trong việc loại bỏ nhiễu ngẫu nhiên (ví dụ: nhiễu muối tiêu)
    bằng cách thay thế giá trị của mỗi pixel bằng giá trị trung vị của các pixel lân cận.

    Args:
        image (np.ndarray): Hình ảnh đầu vào (định dạng BGR của OpenCV).
        kernel_size (int): Kích thước của kernel (phải là số lẻ và lớn hơn 1, ví dụ: 3, 5, 7).

    Returns:
        numpy.ndarray: Hình ảnh đã được lọc nhiễu.
    """
    # Kích thước kernel cho bộ lọc trung vị phải là một số nguyên dương lẻ.
    if kernel_size % 2 == 0 or kernel_size <= 1:
        raise ValueError("Kích thước kernel (kernel_size) phải là một số lẻ lớn hơn 1.")
        
    # Sử dụng hàm cv2.medianBlur của OpenCV để áp dụng bộ lọc.
    # Hàm này hiệu quả trong việc giảm nhiễu mà ít làm mờ các cạnh hơn mean filter.
    filtered_image = cv2.medianBlur(src=image, ksize=kernel_size)
    
    return filtered_image

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

    plt.figure(figsize=(12, 6))
    plt.suptitle("Ứng dụng Bộ lọc Trung vị (Median Filter)", fontsize=16)

    # Hiển thị ảnh gốc
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title("Ảnh Gốc")
    plt.axis('off')

    # Hiển thị ảnh đã lọc
    plt.subplot(1, 2, 2)
    plt.imshow(filtered_rgb)
    plt.title(f"Ảnh đã lọc (Kernel: {kernel_size}x{kernel_size})")
    plt.axis('off')

    plt.show()

# Điểm khởi đầu của chương trình
if __name__ == "__main__":
    # --- THAM SỐ CẤU HÌNH ---
    # URL mặc định của ảnh FPT University
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"
    
    # Kích thước kernel phải là số lẻ (3, 5, 7,...).
    # Kích thước lớn hơn sẽ làm mịn ảnh nhiều hơn.
    KERNEL_SIZE = 5

    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    original_image = load_image_from_url(IMAGE_URL)

    # Chỉ tiếp tục nếu ảnh được tải thành công
    if original_image is not None:
        print(f"Áp dụng bộ lọc trung vị với kernel size = {KERNEL_SIZE}x{KERNEL_SIZE}...")
        
        try:
            # Áp dụng bộ lọc
            filtered_image = apply_median_filter(original_image, KERNEL_SIZE)
            
            print("Hiển thị kết quả...")
            # Hiển thị ảnh gốc và ảnh đã xử lý
            display_images(original_image, filtered_image, KERNEL_SIZE)
            
        except ValueError as e:
            print(f"Lỗi: {e}")