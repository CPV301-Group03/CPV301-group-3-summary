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
    IMAGE_URL = "https://i.chungta.vn/2020/09/07/photo-1-1578027492964533493668-4611-1599472131.jpg" # Ảnh có nhiễu sọc chéo

    print("Đang tải ảnh và xử lý...")
    image = load_image_from_url(IMAGE_URL)

    if image is not None:
        # 1. Chuyển đổi sang miền tần số
        dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Tính phổ tần số để hiển thị
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)

        # 2. Tạo mặt nạ để loại bỏ nhiễu
        # Nhiễu sọc chéo tạo ra các điểm sáng trên phổ tần số.
        # Ta sẽ "che" các điểm sáng này đi.
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Tạo một bản sao của phổ đã dịch chuyển để lọc
        fshift_filtered = dft_shift.copy()

        # Che các điểm nhiễu (tọa độ này được xác định bằng cách quan sát phổ)
        # Đây là một bộ lọc Notch (Notch Filter) thủ công
        cv2.rectangle(fshift_filtered, (ccol-5, crow-90), (ccol+5, crow-75), (0,0,0), -1)
        cv2.rectangle(fshift_filtered, (ccol-5, crow+75), (ccol+5, crow+90), (0,0,0), -1)

        # Tính phổ tần số sau khi lọc để hiển thị
        filtered_spectrum = 20 * np.log(cv2.magnitude(fshift_filtered[:, :, 0], fshift_filtered[:, :, 1]) + 1)

        # 3. Biến đổi ngược để tái tạo ảnh
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
        reconstructed_image = np.uint8(img_back)

        # 4. Hiển thị toàn bộ quy trình
        plt.figure(figsize=(10, 8))
        plt.suptitle("Application: Image Filtering (Periodic Noise Removal)", fontsize=16)

        plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
        plt.title('Original Image with Noise'), plt.axis('off')

        plt.subplot(2, 2, 3), plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Fourier Spectrum'), plt.axis('off')

        plt.subplot(2, 2, 4), plt.imshow(filtered_spectrum, cmap='gray')
        plt.title('Spectrum after Filtering'), plt.axis('off')
        
        plt.subplot(2, 2, 2), plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Reconstructed Image'), plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()