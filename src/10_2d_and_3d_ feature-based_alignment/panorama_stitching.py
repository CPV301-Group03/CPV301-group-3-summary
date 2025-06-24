# panorama_stitching.py

import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

def stitch_images(img1, img2):
    """
    Ghép hai ảnh lại với nhau để tạo ảnh panorama.
    """
    # 1. Phát hiện và mô tả đặc trưng (sử dụng ORB)
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 2. So khớp đặc trưng
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # Sắp xếp các cặp so khớp theo độ tương đồng
    matches = sorted(matches, key=lambda x: x.distance)

    # Giữ lại 15% các cặp so khớp tốt nhất
    num_good_matches = int(len(matches) * 0.15)
    matches = matches[:num_good_matches]

    # Vẽ các cặp so khớp để kiểm tra
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 3. Ước tính Homography bằng RANSAC
    # Lấy tọa độ của các cặp điểm so khớp tốt
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Tìm ma trận Homography
    # RANSAC giúp loại bỏ các outliers (cặp so khớp sai)
    # reprojThresh là ngưỡng pixel tối đa để một điểm được coi là "inlier"
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 4. Warp ảnh thứ nhất để căn chỉnh với ảnh thứ hai
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    # Kích thước của ảnh panorama cuối cùng sẽ là chiều rộng của cả hai ảnh cộng lại
    warped_img1 = cv2.warpPerspective(img1, M, (w1 + w2, h2))

    # 5. Ghép ảnh đã warp với ảnh thứ hai
    # Đặt ảnh thứ hai vào phần bên trái của ảnh panorama trống
    panorama = warped_img1.copy()
    panorama[0:h2, 0:w2] = img2
    
    return panorama, img_matches

def crop_black_borders(image):
    """Cắt bỏ các viền đen thừa trong ảnh panorama."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return image[y:y+h, x:x+w]
    return image


if __name__ == "__main__":
    # Hai ảnh của một tòa nhà chụp từ các góc khác nhau
    URL1 = "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat1.jpg"
    URL2 = "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/stitching/test/data/boat2.jpg"
    
    print("Đang tải các ảnh...")
    image1 = load_image_from_url(URL1)
    image2 = load_image_from_url(URL2)

    if image1 is not None and image2 is not None:
        print("Đang thực hiện ghép ảnh...")
        panorama_result, matches_visualization = stitch_images(image2, image1) # Ghép ảnh 1 vào ảnh 2

        # Cắt bỏ các viền đen
        final_panorama = crop_black_borders(panorama_result)
        
        # --- Hiển thị kết quả ---
        plt.figure(figsize=(20, 10))
        plt.suptitle("Tạo ảnh Panorama bằng Feature-Based Alignment", fontsize=20)

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(matches_visualization, cv2.COLOR_BGR2RGB))
        plt.title("Các cặp đặc trưng được so khớp (Feature Matching)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(final_panorama, cv2.COLOR_BGR2RGB))
        plt.title("Kết quả Panorama")
        plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()