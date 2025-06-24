# warping_demo.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL và chuyển thành định dạng BGR của OpenCV."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi: {e}")
        return None

def forward_warp(src_img, M):
    """
    Thực hiện Forward Warping một cách thủ công.
    
    Args:
        src_img: Ảnh gốc.
        M: Ma trận biến đổi 2x3 (ví dụ: từ cv2.getRotationMatrix2D).

    Returns:
        Ảnh đích sau khi biến đổi.
    """
    rows, cols, _ = src_img.shape
    # Tạo một ảnh đích trống (màu đen)
    dst_img = np.zeros_like(src_img)

    # Lặp qua TỪNG PIXEL của ảnh GỐC
    for y_src in range(rows):
        for x_src in range(cols):
            # Lấy màu của pixel gốc
            pixel_color = src_img[y_src, x_src]
            
            # Tạo vector tọa độ (x, y, 1) để nhân ma trận
            src_pt = np.array([x_src, y_src, 1])
            
            # Tính tọa độ mới trên ảnh đích
            # x' = M11*x + M12*y + M13
            # y' = M21*x + M22*y + M23
            new_pt = M @ src_pt
            x_dst, y_dst = new_pt[0], new_pt[1]
            
            # Vấn đề cốt lõi của Forward Warping:
            # Tọa độ mới (x_dst, y_dst) có thể là số thực.
            # Ta phải làm tròn để gán vào pixel trên ảnh đích.
            # Việc này gây ra "lỗ hổng" vì một số pixel đích có thể không bao giờ được gán giá trị.
            x_dst_int, y_dst_int = int(round(x_dst)), int(round(y_dst))

            # Kiểm tra xem tọa độ mới có nằm trong ảnh đích không
            if 0 <= x_dst_int < cols and 0 <= y_dst_int < rows:
                dst_img[y_dst_int, x_dst_int] = pixel_color
                
    return dst_img

def inverse_warp(src_img, M):
    """
    Thực hiện Inverse Warping bằng hàm có sẵn của OpenCV.
    Đây là phương pháp chuẩn, hiệu quả và không tạo lỗ hổng.
    """
    rows, cols, _ = src_img.shape
    # cv2.warpAffine đã được xây dựng trên nguyên tắc Inverse Warping.
    # Nó lặp qua ảnh đích, tính ngược tọa độ trên ảnh gốc và nội suy màu.
    dst_img = cv2.warpAffine(src_img, M, (cols, rows))
    return dst_img

if __name__ == "__main__":
    IMAGE_URL = "https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    
    print("Đang tải ảnh...")
    image = load_image_from_url(IMAGE_URL)
    
    if image is not None:
        rows, cols, _ = image.shape
        center = (cols / 2, rows / 2)
        angle = 30 # Một góc xoay đủ để thấy rõ vấn đề
        scale = 1.0
        
        # Lấy ma trận xoay
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # --- Thực hiện các phép biến đổi ---
        print("Đang thực hiện Forward Warping (thủ công)...")
        forward_warped_img = forward_warp(image, M)

        print("Đang thực hiện Inverse Warping (dùng OpenCV)...")
        inverse_warped_img = inverse_warp(image, M)

        # --- Hiển thị kết quả để so sánh ---
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        forward_rgb = cv2.cvtColor(forward_warped_img, cv2.COLOR_BGR2RGB)
        inverse_rgb = cv2.cvtColor(inverse_warped_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(18, 6))
        plt.suptitle("So sánh Forward Warping và Inverse Warping", fontsize=16)

        plt.subplot(1, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Ảnh Gốc")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(forward_rgb)
        plt.title("Forward Warping (Thủ công)\n(Xuất hiện lỗ hổng/vết nứt)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(inverse_rgb)
        plt.title("Inverse Warping (Chuẩn)\n(Mịn và không có lỗ hổng)")
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()