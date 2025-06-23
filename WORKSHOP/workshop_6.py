import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Listbox, Scrollbar
from PIL import Image, ImageTk
import os

class MultiImageStitcherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Workshop_6 : Multi-Image Stitching (v2 - Simplified)")
        self.root.geometry("1100x800")

        # --- BIẾN LƯU TRỮ DỮ LIỆU ---
        # Đây là các biến quan trọng cho thuật toán và giao diện
        self.image_paths = []           # Danh sách đường dẫn các ảnh gốc
        self.stitched_image = None      # Ảnh panorama cuối cùng (không có viền)
        self.border_image = None        # Ảnh panorama có vẽ các đường viền
        self.final_contours = []        # Danh sách các đường viền (contours) của từng ảnh gốc trên ảnh panorama
        
        # Biến trạng thái cho Checkbutton
        self.show_borders_var = tk.BooleanVar(value=False)

        # --- GIAO DIỆN NGƯỜI DÙNG (GUI) ---
        # Phần này giữ nguyên, không thay đổi về logic thuật toán
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)

        image_frame = tk.Frame(main_frame)
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

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

        self.chk_borders = ttk.Checkbutton(control_frame, text="Hiện đường viền", variable=self.show_borders_var, command=self.update_display, state=tk.DISABLED)
        self.chk_borders.pack(pady=5)
        
        self.btn_save = ttk.Button(control_frame, text="Lưu kết quả", command=self.save_result, state=tk.DISABLED)
        self.btn_save.pack(fill=tk.X, pady=5)

        self.status_label = ttk.Label(control_frame, text="Sẵn sàng", relief=tk.SUNKEN, anchor=tk.W, wraplength=280)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, ipady=5)

        self.result_image_label = ttk.Label(image_frame, text="Kết quả sẽ hiển thị ở đây", relief=tk.SOLID, anchor=tk.CENTER)
        self.result_image_label.pack(fill=tk.BOTH, expand=True)

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN GIAO DIỆN ---
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
        self.btn_stitch.config(state=tk.NORMAL if len(self.image_paths) >= 2 else tk.DISABLED)

    def update_display(self):
        if self.stitched_image is None: return
        image_to_show = self.border_image if self.show_borders_var.get() and self.border_image is not None else self.stitched_image
        self.display_result_image(image_to_show)

    def display_result_image(self, cv_image):
        if cv_image is None: return
        # Resize ảnh để vừa với cửa sổ hiển thị
        widget_w = self.result_image_label.winfo_width()
        widget_h = self.result_image_label.winfo_height()
        if widget_w < 50 or widget_h < 50: widget_w, widget_h = 800, 700
        img_h, img_w = cv_image.shape[:2]
        ratio = min(widget_w / img_w, widget_h / img_h)
        new_size = (int(img_w * ratio), int(img_h * ratio))
        
        thumbnail = cv2.resize(cv_image, new_size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tk_img = ImageTk.PhotoImage(image=pil_img)
        
        self.result_image_label.config(image=tk_img)
        self.result_image_label.image = tk_img

    def save_result(self):
        if self.stitched_image is None: return
        image_to_save = self.border_image if self.show_borders_var.get() and self.border_image is not None else self.stitched_image
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if save_path:
            cv2.imwrite(save_path, image_to_save)
            messagebox.showinfo("Thành công", f"Đã lưu kết quả tại:\n{save_path}")

    # --- CÁC HÀM CỐT LÕI CỦA THUẬT TOÁN GHÉP ẢNH ---

    def perform_stitching(self):
        """
        Hàm chính điều khiển toàn bộ quá trình ghép ảnh.
        """
        if len(self.image_paths) < 2: return
        
        self.status_label.config(text="Đang đọc các file ảnh...")
        self.root.update_idletasks()
        
        images = [cv2.imread(path) for path in self.image_paths]

        # Gọi hàm thuật toán chính để ghép nhiều ảnh
        result_pano, self.final_contours = self.stitch_multiple_images(images)

        if result_pano is None:
            messagebox.showerror("Thất bại", "Quá trình ghép ảnh thất bại. Có thể do các ảnh không có đủ điểm chung.")
            self.status_label.config(text="Ghép ảnh thất bại.")
        else:
            self.stitched_image = result_pano
            self.create_border_image() # Tạo ảnh có viền sau khi đã có ảnh panorama
            
            # Cập nhật giao diện sau khi thành công
            self.btn_save.config(state=tk.NORMAL)
            self.chk_borders.config(state=tk.NORMAL)
            self.status_label.config(text="Ghép ảnh thành công!")
            self.update_display()

    def stitch_multiple_images(self, images):
        """
        Thuật toán ghép một danh sách các ảnh theo thứ tự.
        Nó hoạt động bằng cách ghép lặp đi lặp lại: (ảnh 2 vào ảnh 1), rồi (ảnh 3 vào kết quả), ...
        """
        num_images = len(images)
        
        # Bắt đầu với ảnh đầu tiên làm panorama ban đầu
        current_panorama = images[0]
        
        # Danh sách này sẽ lưu các ma trận homography cuối cùng.
        # Mỗi ma trận H[i] sẽ biến đổi ảnh gốc images[i] vào đúng vị trí trên panorama cuối cùng.
        # Ma trận đầu tiên là ma trận đơn vị vì ảnh đầu tiên là gốc.
        final_homographies = [np.identity(3)]

        # Lặp qua các ảnh còn lại để ghép vào panorama
        for i in range(1, num_images):
            next_image = images[i]
            
            self.status_label.config(text=f"Đang ghép ảnh {i+1}/{num_images}...")
            self.root.update_idletasks()
            
            # Ghép ảnh tiếp theo (bên phải) vào panorama hiện tại (bên trái)
            # Hàm này trả về 3 giá trị quan trọng:
            # 1. new_pano: Ảnh panorama mới đã được mở rộng.
            # 2. H_pano_to_new: Ma trận để "vẽ" panorama CŨ vào vị trí mới trên new_pano.
            # 3. H_img_to_new: Ma trận để "vẽ" ảnh MỚI vào vị trí mới trên new_pano.
            new_pano, H_pano_to_new, H_img_to_new = self.stitch_pair(current_panorama, next_image)
            
            if new_pano is None:
                return None, None # Ghép thất bại, dừng lại
            
            # CẬP NHẬT TRẠNG THÁI CHO VÒNG LẶP TIẾP THEO
            # 1. Cập nhật panorama
            current_panorama = new_pano
            
            # 2. Cập nhật ma trận homography của TẤT CẢ các ảnh đã được ghép TRƯỚC ĐÓ.
            # Chúng cần được di chuyển theo ma trận H_pano_to_new.
            for j in range(len(final_homographies)):
                # Dùng toán tử @ cho phép nhân ma trận (tương đương np.dot)
                final_homographies[j] = H_pano_to_new @ final_homographies[j]
            
            # 3. Thêm ma trận của ảnh vừa được ghép vào danh sách.
            final_homographies.append(H_img_to_new)

        # --- TÍNH TOÁN ĐƯỜNG VIỀN CUỐI CÙNG ---
        # Sau khi có panorama cuối cùng và tất cả các ma trận biến đổi, ta tìm đường viền.
        final_contours = []
        for i in range(num_images):
            original_image = images[i]
            h, w = original_image.shape[:2]
            
            # Tạo một hình chữ nhật bao quanh ảnh gốc.
            corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
            # Áp dụng ma trận homography cuối cùng để tìm vị trí mới của các góc này
            # trên ảnh panorama cuối cùng.
            transformed_corners = cv2.perspectiveTransform(corners, final_homographies[i])
            
            final_contours.append(np.int32(transformed_corners))
            
        return current_panorama, final_contours

    def stitch_pair(self, img_left, img_right):
        """
        Hàm cốt lõi: Ghép một cặp ảnh (img_left và img_right).
        Trả về ảnh panorama mới và các ma trận biến đổi cần thiết.
        """
        # --- BƯỚC 1: TÌM CÁC ĐIỂM TƯƠNG ĐỒNG (KEYPOINTS) VÀ TÍNH HOMOGRAPHY ---
        orb = cv2.ORB_create(nfeatures=2000)
        keypoints_left, descriptors_left = orb.detectAndCompute(img_left, None)
        keypoints_right, descriptors_right = orb.detectAndCompute(img_right, None)

        if descriptors_left is None or descriptors_right is None: return None, None, None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = matcher.match(descriptors_right, descriptors_left)
        except cv2.error:
            return None, None, None

        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:int(len(matches) * 0.20)]
        
        if len(good_matches) < 10:
            return None, None, None

        src_pts = np.float32([keypoints_right[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_left[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M_right_to_left, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M_right_to_left is None:
            return None, None, None

        # --- BƯỚC 2: TÍNH KÍCH THƯỚC CHO KHUNG ẢNH PANORAMA MỚI ---
        h_left, w_left = img_left.shape[:2]
        h_right, w_right = img_right.shape[:2]

        corners_left = np.float32([[0, 0], [0, h_left], [w_left, h_left], [w_left, 0]]).reshape(-1, 1, 2)
        corners_right = np.float32([[0, 0], [0, h_right], [w_right, h_right], [w_right, 0]]).reshape(-1, 1, 2)
        corners_right_transformed = cv2.perspectiveTransform(corners_right, M_right_to_left)
        
        all_corners = np.concatenate((corners_left, corners_right_transformed), axis=0)
        
        x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
        x_max, y_max = np.int32(all_corners.max(axis=0).ravel())
        
        output_width = x_max - x_min
        output_height = y_max - y_min

        # --- BƯỚC 3: TẠO MA TRẬN BIẾN ĐỔI CUỐI CÙNG VÀ WARP ẢNH ---
        H_translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        H_left_final = H_translation
        H_right_final = H_translation @ M_right_to_left

        warped_left = cv2.warpPerspective(img_left, H_left_final, (output_width, output_height))
        warped_right = cv2.warpPerspective(img_right, H_right_final, (output_width, output_height))

        # --- BƯỚC 4: TRỘN (BLENDING) HAI ẢNH Ở VÙNG GIAO NHAU ---
        result_pano = warped_left.copy()
        
        gray_right_warped = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)
        _, mask_right_pixels = cv2.threshold(gray_right_warped, 0, 255, cv2.THRESH_BINARY)
        result_pano[mask_right_pixels > 0] = warped_right[mask_right_pixels > 0]

        gray_left = cv2.cvtColor(warped_left, cv2.COLOR_BGR2GRAY)
        _, mask_left = cv2.threshold(gray_left, 0, 255, cv2.THRESH_BINARY)
        
        gray_right = cv2.cvtColor(warped_right, cv2.COLOR_BGR2GRAY)
        _, mask_right = cv2.threshold(gray_right, 0, 255, cv2.THRESH_BINARY)
        
        overlap_mask = cv2.bitwise_and(mask_left, mask_right)
        
        rows, cols = np.where(overlap_mask > 0)
        if len(rows) > 0:
            y_min_roi, y_max_roi = np.min(rows), np.max(rows)
            x_min_roi, x_max_roi = np.min(cols), np.max(cols)
            
            left_roi = warped_left[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            right_roi = warped_right[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
            
            h_roi, w_roi = left_roi.shape[:2]
            if w_roi > 0:
                alpha = np.linspace(1.0, 0.0, w_roi)
                alpha_gradient = np.tile(alpha, (h_roi, 1))
                alpha_gradient_3ch = cv2.cvtColor(alpha_gradient.astype(np.float32), cv2.COLOR_GRAY2BGR)
                
                # === SỬA LỖI Ở ĐÂY ===
                # Thay vì dùng cv2.addWeighted, ta thực hiện phép toán trộn trực tiếp bằng NumPy.
                # Phép toán này là element-wise (từng pixel nhân với trọng số tương ứng của nó).
                blended_roi = (left_roi.astype(np.float32) * alpha_gradient_3ch) + \
                              (right_roi.astype(np.float32) * (1.0 - alpha_gradient_3ch))
                
                target_roi = result_pano[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
                overlap_roi_mask = overlap_mask[y_min_roi:y_max_roi+1, x_min_roi:x_max_roi+1]
                # Chuyển đổi blended_roi về uint8 trước khi copy
                np.copyto(target_roi, blended_roi.astype(np.uint8), where=np.repeat(overlap_roi_mask[:, :, np.newaxis], 3, axis=2) > 0)

        # --- BƯỚC 5: CẮT BỎ VIỀN ĐEN VÀ TRẢ VỀ KẾT QUẢ ---
        final_pano = self.crop_black_border(result_pano)
        
        return final_pano, H_left_final, H_right_final

    def create_border_image(self):
        """Tạo một bản sao của ảnh panorama và vẽ các đường viền lên đó."""
        if self.stitched_image is None or not self.final_contours:
            self.border_image = None
            return
        
        self.border_image = self.stitched_image.copy()
        # Dùng các màu khác nhau để vẽ viền cho dễ phân biệt
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        
        for i, contour in enumerate(self.final_contours):
            if contour is not None:
                color = colors[i % len(colors)]
                cv2.drawContours(self.border_image, [contour], -1, color, 3) # Vẽ đường viền dày hơn
        
    def crop_black_border(self, image):
        """Cắt bỏ các vùng viền đen thừa xung quanh ảnh panorama."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Tìm contour lớn nhất (chính là vùng ảnh hợp lệ)
            c = max(contours, key=cv2.contourArea)
            # Lấy hình chữ nhật bao quanh contour đó
            x, y, w, h = cv2.boundingRect(c)
            # Cắt ảnh theo hình chữ nhật này
            return image[y:y + h, x:x + w]
        return image
    
if __name__ == "__main__":
    root = tk.Tk()
    app = MultiImageStitcherApp(root)
    root.mainloop()