import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Listbox, Scrollbar
from PIL import Image, ImageTk
import os

class MultiImageStitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Workshop #2: Multi-Image Stitching (v2)")
        self.root.geometry("1100x800")

        # Biến lưu trữ dữ liệu
        self.image_paths = []
        self.stitched_image = None
        self.border_image = None # Ảnh mới để lưu phiên bản có đường viền
        self.final_contours = [] # Lưu các đường viền cuối cùng
        
        # Biến Tkinter cho Checkbutton
        self.show_borders_var = tk.BooleanVar(value=False)

        # --- Giao diện ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Frame điều khiển (bên trái)
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        # Frame hiển thị ảnh (bên phải)
        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Các thành phần trong Frame điều khiển ---
        ttk.Label(control_frame, text="Điều khiển", font=("Helvetica", 16, "bold")).pack(pady=10)

        self.btn_load = ttk.Button(control_frame, text="Thêm ảnh (theo thứ tự)", command=self.load_images)
        self.btn_load.pack(fill=tk.X, pady=5)

        self.btn_clear = ttk.Button(control_frame, text="Xóa danh sách", command=self.clear_list)
        self.btn_clear.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Danh sách ảnh (trái -> phải):").pack(pady=(10, 0), anchor=tk.W)
        list_frame = tk.Frame(control_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.listbox = Listbox(list_frame, selectmode=tk.SINGLE)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.config(yscrollcommand=scrollbar.set)
        
        self.btn_stitch = ttk.Button(control_frame, text="Ghép tất cả ảnh", command=self.perform_stitching, state=tk.DISABLED)
        self.btn_stitch.pack(fill=tk.X, ipady=10, pady=10)

        # === THAY ĐỔI 1: THÊM CHECKBUTTON ===
        self.chk_borders = ttk.Checkbutton(control_frame, text="Hiện đường viền", variable=self.show_borders_var, command=self.update_display, state=tk.DISABLED)
        self.chk_borders.pack(pady=5)
        
        self.btn_save = ttk.Button(control_frame, text="Lưu kết quả", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(control_frame, text="Sẵn sàng", relief=tk.SUNKEN, anchor=tk.W, wraplength=280)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=5)

        self.result_image_label = ttk.Label(image_frame, text="Kết quả sẽ hiển thị ở đây", relief=tk.SOLID, anchor=tk.CENTER)
        self.result_image_label.pack(fill=tk.BOTH, expand=True)

    def load_images(self):
        paths = filedialog.askopenfilenames(title="Chọn các ảnh theo thứ tự từ trái sang phải", filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if paths:
            for path in paths:
                if path not in self.image_paths:
                    self.image_paths.append(path)
                    self.listbox.insert(tk.END, os.path.basename(path))
            self.update_stitch_button_state()
            self.status_label.config(text=f"Đã thêm {len(paths)} ảnh. Tổng cộng: {len(self.image_paths)} ảnh.")

    def clear_list(self):
        self.image_paths.clear()
        self.listbox.delete(0, tk.END)
        self.stitched_image = None
        self.border_image = None
        self.final_contours = []
        self.result_image_label.config(image='', text="Kết quả sẽ hiển thị ở đây")
        self.update_stitch_button_state()
        self.btn_save.config(state=tk.DISABLED)
        self.chk_borders.config(state=tk.DISABLED)
        self.show_borders_var.set(False)
        self.status_label.config(text="Đã xóa danh sách.")

    def update_stitch_button_state(self):
        if len(self.image_paths) >= 2:
            self.btn_stitch.config(state=tk.NORMAL)
        else:
            self.btn_stitch.config(state=tk.DISABLED)

    # === THAY ĐỔI 2: TẠO HÀM UPDATE RIÊNG ===
    def update_display(self):
        """Hàm này quyết định sẽ hiển thị ảnh nào dựa trên checkbox."""
        if self.stitched_image is None:
            return
            
        if self.show_borders_var.get() and self.border_image is not None:
            self.display_result_image(self.border_image)
            self.status_label.config(text="Hiển thị ảnh với đường viền.")
        else:
            self.display_result_image(self.stitched_image)
            self.status_label.config(text="Hiển thị ảnh đã ghép.")

    def display_result_image(self, cv_image):
        if cv_image is None: return
        widget_w = self.result_image_label.winfo_width()
        widget_h = self.result_image_label.winfo_height()
        if widget_w < 50 or widget_h < 50: widget_w, widget_h = 800, 700
        img_h, img_w = cv_image.shape[:2]
        ratio = min(widget_w / img_w, widget_h / img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        thumbnail = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        self.result_image_label.config(image=img_tk)
        self.result_image_label.image = img_tk

    def perform_stitching(self):
        if len(self.image_paths) < 2: return
        
        self.final_contours = [] # Reset danh sách contours
        images = [cv2.imread(path) for path in self.image_paths]
        
        result, contours = self.stitch_multiple_images(images)

        if result is None:
            messagebox.showerror("Thất bại", "Quá trình ghép ảnh thất bại.")
        else:
            self.stitched_image = result
            self.final_contours = contours
            self.create_border_image() # Tạo ảnh có đường viền
            self.btn_save.config(state=tk.NORMAL)
            self.chk_borders.config(state=tk.NORMAL) # Kích hoạt checkbox
            self.status_label.config(text="Ghép ảnh thành công!")
            self.update_display() # Hiển thị kết quả

    # === THAY ĐỔI 3: HÀM VẼ ĐƯỜNG VIỀN ===
    def create_border_image(self):
        """Tạo một bản sao của ảnh đã ghép và vẽ các đường viền lên đó."""
        if self.stitched_image is None or not self.final_contours:
            return
        
        self.border_image = self.stitched_image.copy()
        # Danh sách màu sắc để các đường viền khác nhau
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, contour in enumerate(self.final_contours):
            color = colors[i % len(colors)]
            cv2.drawContours(self.border_image, [contour], -1, color, 3) # Vẽ với độ dày là 3

    # === THAY ĐỔI 4: SỬA HÀM GHÉP NHIỀU ẢNH ===
    def stitch_multiple_images(self, image_list):
        panorama = image_list[0]
        all_contours = []

        for i in range(1, len(image_list)):
            next_image = image_list[i]
            self.status_label.config(text=f"Đang ghép ảnh {i+1}/{len(image_list)}...")
            self.root.update_idletasks()
            
            # Hàm stitch_images giờ sẽ trả về cả các contour
            new_panorama, left_contour, right_contour = self.stitch_images(next_image, panorama)
            
            if new_panorama is None:
                self.status_label.config(text=f"Ghép ảnh {i+1} thất bại. Dừng lại.")
                return None, None
            
            panorama = new_panorama
            
            if i == 1: # Ở lần lặp đầu tiên, thêm contour của cả ảnh trái và phải
                all_contours.append(left_contour)
            all_contours.append(right_contour)
        
        return panorama, all_contours

    # === THAY ĐỔI 5: SỬA HÀM GHÉP 2 ẢNH ĐỂ TRẢ VỀ CONTOUR ===
    def stitch_images(self, img_right, img_left):
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)
        if descriptors_left is None or descriptors_right is None: return None, None, None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(matcher.match(descriptors_right, descriptors_left), key=lambda x: x.distance)
        matches = matches[:int(len(matches) * 0.20)]
        if len(matches) < 10: return None, None, None
        src_pts = np.float32([keypoints_right[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_left[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None: return None, None, None

        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]
        corners_left = np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]).reshape(-1, 1, 2)
        corners_right_transformed = cv2.perspectiveTransform(np.float32([[0, 0], [0, h_right], [w_right, h_right], [w_right, 0]]).reshape(-1, 1, 2), M)
        all_corners = np.concatenate((corners_left, corners_right_transformed), axis=0)
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        output_width = x_max - x_min
        output_height = y_max - y_min

        warped_left = cv2.warpPerspective(img_left, H_translation, (output_width, output_height))
        warped_right = cv2.warpPerspective(img_right, H_translation.dot(M), (output_width, output_height))
        
        # --- Lấy contour của mỗi ảnh đã được warp ---
        contour_left = self.get_contour_from_image(warped_left)
        contour_right = self.get_contour_from_image(warped_right)

        # Logic blending giữ nguyên như trước
        result_pano = warped_left.copy()
        gray_right = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)
        _, mask_right = cv2.threshold(gray_right, 0, 255, cv2.THRESH_BINARY)
        result_pano[mask_right > 0] = warped_right[mask_right > 0]
        gray_left = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY)
        _, mask_left = cv2.threshold(gray_left, 0, 255, cv2.THRESH_BINARY)
        overlap_mask = cv2.bitwise_and(mask_left, mask_right)
        rows, cols = np.where(overlap_mask > 0)
        if len(rows) > 0 and len(cols) > 0:
            y_min_roi, y_max_roi = np.min(rows), np.max(rows)
            x_min_roi, x_max_roi = np.min(cols), np.max(cols)
            left_roi = warped_left[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            right_roi = warped_right[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            overlap_roi_mask = overlap_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            roi_height, roi_width = left_roi.shape[:2]
            if roi_width > 0:
                alpha_gradient = np.tile(np.linspace(1.0, 0.0, roi_width), (roi_height, 1))
                alpha_gradient_3ch = cv2.cvtColor(alpha_gradient.astype(np.float32), cv2.COLOR_GRAY2BGR)
                blended_roi = (left_roi.astype(np.float32) * alpha_gradient_3ch + right_roi.astype(np.float32) * (1.0 - alpha_gradient_3ch)).astype(np.uint8)
                target_roi = result_pano[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
                where_mask = np.repeat(overlap_roi_mask[:, :, np.newaxis], 3, axis=2) > 0
                np.copyto(target_roi, blended_roi, where=where_mask)
        
        final_pano = self.crop_black_border(result_pano)

        # Trả về cả panorama và các contour
        return final_pano, contour_left, contour_right

    def get_contour_from_image(self, image):
        """Hàm phụ để tìm contour lớn nhất từ một ảnh."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            return max(contours, key=cv2.contourArea)
        return None
        
    def crop_black_border(self, image):
        contour = self.get_contour_from_image(image)
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            return image[y:y + h, x:x + w]
        return image

    def save_result(self):
        if self.stitched_image is None: return
        # Hỏi người dùng muốn lưu ảnh nào
        image_to_save = self.border_image if self.show_borders_var.get() else self.stitched_image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, image_to_save)
            messagebox.showinfo("Thành công", f"Đã lưu kết quả tại:\n{save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiImageStitcherApp(root)
    root.mainloop()