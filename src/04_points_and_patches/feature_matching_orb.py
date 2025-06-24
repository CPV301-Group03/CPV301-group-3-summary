# feature_matching_orb.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str, grayscale=False):
    """Tải ảnh từ URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imdecode(image_array, mode)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    # Hai ảnh mẫu: một ảnh là vật thể, ảnh kia là cảnh chứa vật thể đó.
    URL_OBJECT = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box.png"
    URL_SCENE = "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/box_in_scene.png"

    print("Đang tải ảnh...")
    img1_gray = load_image_from_url(URL_OBJECT, grayscale=True) # Ảnh vật thể (query)
    img2_gray = load_image_from_url(URL_SCENE, grayscale=True)  # Ảnh cảnh (train)

    # Tải lại ảnh màu để vẽ
    img1_color = load_image_from_url(URL_OBJECT)
    img2_color = load_image_from_url(URL_SCENE)

    if img1_gray is not None and img2_gray is not None:
        # 1. Khởi tạo ORB detector
        # ORB là một giải pháp thay thế hiệu quả cho SIFT/SURF
        orb = cv2.ORB_create(nfeatures=1000)

        # 2. Tìm keypoints và descriptors với ORB
        print("Phát hiện và mô tả các điểm đặc trưng...")
        kp1, des1 = orb.detectAndCompute(img1_gray, None)
        kp2, des2 = orb.detectAndCompute(img2_gray, None)

        # 3. Tạo đối tượng Brute-Force Matcher
        # NORM_HAMMING được dùng cho binary descriptors như của ORB, BRIEF
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # 4. So khớp các descriptors
        print("So khớp các điểm đặc trưng...")
        matches = bf.match(des1, des2)

        # 5. Sắp xếp các cặp so khớp theo khoảng cách (distance)
        # Khoảng cách càng nhỏ, cặp so khớp càng tốt.
        matches = sorted(matches, key=lambda x: x.distance)

        # 6. Vẽ N cặp so khớp tốt nhất
        num_good_matches = 50
        img_matches = cv2.drawMatches(
            img1_color, kp1, 
            img2_color, kp2, 
            matches[:num_good_matches], 
            None, 
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # 7. Hiển thị kết quả
        print("Hiển thị kết quả so khớp.")
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matching using ORB (Top {num_good_matches} matches)")
        plt.axis('off')
        plt.show()