# geometric_transformations_demo.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải một hình ảnh từ URL và chuyển đổi nó thành định dạng BGR của OpenCV."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError("Không thể giải mã hình ảnh từ URL.")
        return image
    except (requests.exceptions.RequestException, IOError) as e:
        print(f"Lỗi: {e}")
        return None

if __name__ == "__main__":
    # Sử dụng ảnh ngôi nhà tương tự như trong slide
    IMAGE_URL = "https://images.pexels.com/photos/186077/pexels-photo-186077.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1"
    
    print(f"Đang tải ảnh từ: {IMAGE_URL}")
    image = load_image_from_url(IMAGE_URL)
    
    if image is not None:
        rows, cols, _ = image.shape

        # --- 1. Translation (Tịnh tiến) ---
        # Ma trận tịnh tiến: di chuyển ảnh sang phải 100px và xuống dưới 50px
        M_trans = np.float32([[1, 0, 100], [0, 1, 50]])
        img_trans = cv2.warpAffine(image, M_trans, (cols, rows))

        # --- 2. Rotation (Xoay) ---
        # Tạo ma trận xoay: xoay 45 độ quanh tâm ảnh, không co giãn
        center = (cols / 2, rows / 2)
        M_rot = cv2.getRotationMatrix2D(center, angle=45, scale=1)
        img_rot = cv2.warpAffine(image, M_rot, (cols, rows))

        # --- 3. Affine Transform (Biến dạng xiên - Shear) ---
        # Chọn 3 điểm trên ảnh gốc và vị trí mới của chúng trên ảnh đích
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[10, 100], [200, 50], [100, 250]]) # Vị trí mới tạo hiệu ứng xiên
        M_affine = cv2.getAffineTransform(pts1, pts2)
        img_affine = cv2.warpAffine(image, M_affine, (cols, rows))

        # --- 4. Scaling (Co giãn - thuộc nhóm Similarity) ---
        # Co nhỏ ảnh lại còn 75% kích thước
        # Có thể dùng cv2.resize hoặc warpAffine với getRotationMatrix2D
        M_scale = cv2.getRotationMatrix2D(center, angle=0, scale=0.75)
        img_scale = cv2.warpAffine(image, M_scale, (cols, rows))
        
        # --- 5. Perspective Transform (Biến đổi phối cảnh) ---
        # Chọn 4 điểm trên ảnh gốc (4 góc) và vị trí mới của chúng
        pts1_pers = np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]])
        # Tạo hiệu ứng nhìn từ trên xuống, làm cho cạnh trên ngắn lại
        pts2_pers = np.float32([[100, 50], [cols - 100, 50], [0, rows - 1], [cols - 1, rows - 1]])
        M_pers = cv2.getPerspectiveTransform(pts1_pers, pts2_pers)
        img_pers = cv2.warpPerspective(image, M_pers, (cols, rows))
        
        # --- Hiển thị kết quả ---
        # Chuyển đổi BGR sang RGB để Matplotlib hiển thị đúng màu
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transformations = {
            "Translation": cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB),
            "Rotation (Euclidean)": cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB),
            "Scaling (Similarity)": cv2.cvtColor(img_scale, cv2.COLOR_BGR2RGB),
            "Affine (Shear)": cv2.cvtColor(img_affine, cv2.COLOR_BGR2RGB),
            "Projective": cv2.cvtColor(img_pers, cv2.COLOR_BGR2RGB),
        }
        
        plt.figure(figsize=(15, 10))
        plt.suptitle("Geometric Transformations Demo", fontsize=16)

        # Hiển thị ảnh gốc
        plt.subplot(2, 3, 1)
        plt.imshow(image_rgb)
        plt.title("Original Image")
        plt.axis('off')

        # Hiển thị các ảnh đã biến đổi
        for i, (title, img) in enumerate(transformations.items()):
            plt.subplot(2, 3, i + 2)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()