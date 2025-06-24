# image_reconstruction_from_noise.py
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

        # 2. Tạo mặt nạ lọc thông thấp để giữ lại nội dung chính
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        RADIUS = 60 # Bán kính vùng tần số thấp được giữ lại
        mask = np.zeros((rows, cols, 2), np.uint8)
        cv2.circle(mask, (ccol, crow), RADIUS, (1, 1), -1)
        
        # Chuyển mặt nạ sang ảnh 8-bit để hiển thị
        mask_display = (mask[:, :, 0] * 255).astype(np.uint8)
        
        # 3. Áp dụng mặt nạ
        fshift_filtered = dft_shift * mask

        # 4. Biến đổi ngược để tái tạo ảnh
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        reconstructed_image = np.uint8(img_back)

        # 5. Hiển thị toàn bộ quy trình
        plt.figure(figsize=(10, 8))
        plt.suptitle("Application: Image Reconstruction", fontsize=16)

        plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Original Image with Grid Noise'), plt.axis('off')

        plt.subplot(2, 2, 2), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Fourier Spectrum'), plt.axis('off')

        plt.subplot(2, 2, 3), plt.imshow(mask_display, cmap='gray')
        plt.title('Low-Pass Filter Mask'), plt.axis('off')
        
        plt.subplot(2, 2, 4), plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Reconstructed Image'), plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()