import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Workshop")
        self.image = None

        # Tạo khung chứa 2 ảnh theo hàng ngang
        image_frame = Frame(root)
        image_frame.pack(pady=10)

        Label(image_frame, text="Original Image").grid(row=0, column=0)
        Label(image_frame, text="Processed Image").grid(row=0, column=1)

        self.original_img_display = Label(image_frame)
        self.original_img_display.grid(row=1, column=0, padx=10)

        self.processed_img_display = Label(image_frame)
        self.processed_img_display.grid(row=1, column=1, padx=10)

        # Nút bấm các chức năng
        Button(root, text="📂 Load Image", command=self.load_image).pack(pady=5)
        Button(root, text="1. Color Balance", command=self.color_balance).pack(pady=2)
        Button(root, text="2. Histogram Equalization", command=self.histogram_equalization).pack(pady=2)
        Button(root, text="3. Median Filter", command=self.median_filter).pack(pady=2)
        Button(root, text="4. Mean Filter", command=self.mean_filter).pack(pady=2)
        Button(root, text="5. Gaussian Smoothing", command=self.gaussian_smoothing).pack(pady=2)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if path:
            try:
                file_data = np.fromfile(path, dtype=np.uint8)
                self.image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
                if self.image is None:
                    raise ValueError("Ảnh không hợp lệ hoặc bị hỏng.")
                self.show_image_on_label(self.image, self.original_img_display)
                self.processed_img_display.configure(image='')  # Xóa ảnh cũ
            except Exception as e:
                messagebox.showerror("Lỗi", f"❌ Không thể đọc ảnh:\n{e}")

    def show_image_on_label(self, img, label):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((300, 300))  # Resize nhỏ hơn vì có 2 ảnh
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk

    def display_images(self, original, processed):
        self.show_image_on_label(original, self.original_img_display)
        self.show_image_on_label(processed, self.processed_img_display)

    def color_balance(self):
        if self.image is not None:
            balanced = cv2.convertScaleAbs(self.image, alpha=1.1, beta=10)
            self.display_images(self.image, balanced)

    def histogram_equalization(self):
        if self.image is not None:
            img_yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
            img_eq = img_yuv.copy()
            img_eq[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
            equalized = cv2.cvtColor(img_eq, cv2.COLOR_YUV2BGR)

            # Hiển thị cả 2 biểu đồ histogram
            plt.figure(figsize=(10, 4))

            # Histogram gốc
            plt.subplot(1, 2, 1)
            for i, col in enumerate(('b', 'g', 'r')):
                hist = cv2.calcHist([self.image], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.title("Original Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")

            # Histogram sau khi cân bằng
            plt.subplot(1, 2, 2)
            for i, col in enumerate(('b', 'g', 'r')):
                hist = cv2.calcHist([equalized], [i], None, [256], [0, 256])
                plt.plot(hist, color=col)
            plt.title("Equalized Histogram")
            plt.xlabel("Pixel Value")
            plt.ylabel("Frequency")

            plt.tight_layout()
            plt.show()

            self.display_images(self.image, equalized)

    def median_filter(self):
        if self.image is not None:
            median = cv2.medianBlur(self.image, 5)
            self.display_images(self.image, median)

    def mean_filter(self):
        if self.image is not None:
            mean = cv2.blur(self.image, (5, 5))
            self.display_images(self.image, mean)

    def gaussian_smoothing(self):
        if self.image is not None:
            gaussian = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.display_images(self.image, gaussian)

if __name__ == "__main__":
    root = Tk()
    root.geometry("800x600")
    app = ImageProcessorApp(root)
    root.mainloop()
