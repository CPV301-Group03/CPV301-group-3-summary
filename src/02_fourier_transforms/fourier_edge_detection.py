# edge_detection_with_fourier.py
import cv2
import numpy as np
import requests
import matplotlib.pyplot as plt

def load_image_from_url(url: str):
    """Tải ảnh từ URL và chuyển thành ảnh xám."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
        if image is None: raise IOError("Không thể giải mã ảnh.")
        # Resize để xử lý nhanh hơn và phù hợp hiển thị
        image = cv2.resize(image, (512, int(512 * image.shape[0] / image.shape[1])))
        return image
    except Exception as e:
        print(f"Lỗi tải ảnh: {e}")
        return None

if __name__ == "__main__":
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg"

    print("Đang tải ảnh và xử lý...")
    image = load_image_from_url(IMAGE_URL)

    if image is not None:
        # 1. Chuyển đổi sang miền tần số
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
        
        # 2. Tạo mặt nạ lọc thông cao (High-Pass Filter)
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        RADIUS = 30 # Bán kính vùng tần số thấp bị loại bỏ
        mask = np.ones((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), RADIUS, (0, 0), -1)

        # 3. Áp dụng mặt nạ
        fshift_filtered = dft_shift * mask
        filtered_spectrum = 20 * np.log(cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1)

        # 4. Biến đổi ngược để tái tạo ảnh cạnh
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        edge_image = np.uint8(img_back)

        # 5. Hiển thị toàn bộ quy trình
        plt.figure(figsize=(10, 8))
        plt.suptitle("Application: Edge Detection", fontsize=16)

        plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Input Image'), plt.axis('off')

        plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('After FFT (Spectrum)'), plt.axis('off')

        plt.subplot(2, 2, 3), plt.imshow(filtered_spectrum, cmap='gray')
        plt.title('FFT + Mask (High-Pass)'), plt.axis('off')
        
        plt.subplot(2, 2, 4), plt.imshow(edge_image, cmap='gray')
        plt.title('After FFT Inverse (Edges)'), plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()