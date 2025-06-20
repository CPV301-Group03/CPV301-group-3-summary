import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk

def align_images(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    total_matches = len(matches)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]  #lowe Ratio Test
    good_matches_count = len(good)

    info = {
        "keypoints_img1": len(kp1),
        "keypoints_img2": len(kp2),
        "total_matches": total_matches,
        "good_matches": good_matches_count
    }

    if good_matches_count > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        ransacReprojThreshold = 5.0
        maxIters = 2000 #Rêpat
        confidence = 0.995

        H, mask = cv2.findHomography(
            src_pts, dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold,
            maxIters=maxIters,
            confidence=confidence
        )

        if H is not None and mask is not None:
            inliers = np.sum(mask) #K
            inlier_ratio = inliers / good_matches_count
            info["inliers"] = int(inliers)
            info["inlier_ratio"] = inlier_ratio
        else:
            info["inliers"] = 0
            info["inlier_ratio"] = 0.0

        h, w = img2.shape
        aligned = cv2.warpPerspective(img1, H, (w, h))
        return aligned, info, src_pts, dst_pts, mask
    else:
        return None, info, None, None, None

def draw_matches(aligned_img1, img2, src_pts, dst_pts, mask):
    img1_color = cv2.cvtColor(aligned_img1, cv2.COLOR_GRAY2BGR)
    img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_color
    vis[:h2, w1:w1 + w2] = img2_color

    for i, m in enumerate(mask.ravel()):
        if m:
            pt1 = tuple(np.int32(src_pts[i][0]))
            pt2 = tuple(np.int32(dst_pts[i][0]))
            pt2_shifted = (int(pt2[0] + w1), int(pt2[1]))
            cv2.line(vis, pt1, pt2_shifted, (0, 255, 0), 1)
            cv2.circle(vis, pt1, 3, (255, 0, 0), -1)
            cv2.circle(vis, pt2_shifted, 3, (0, 0, 255), -1)
    return vis

class RANSACApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RANSAC Image Alignment")
        self.root.geometry("1600x800")
        self.img1 = None
        self.img2 = None

        # Buttons
        top_frame = tk.Frame(root)
        top_frame.pack(pady=10)

        tk.Button(top_frame, text="Load Image 1", command=self.load_img1).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Image 2", command=self.load_img2).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Run RANSAC", command=self.run_ransac).pack(side=tk.LEFT, padx=5)

        # Info box
        self.info_text = tk.StringVar()
        tk.Label(root, textvariable=self.info_text, justify="left", font=("Courier", 10)).pack(pady=10)

        # Image display frame
        self.img_frame = tk.Frame(root)
        self.img_frame.pack()

        self.img_labels = []
        for _ in range(4):
            lbl = tk.Label(self.img_frame)
            lbl.pack(side=tk.LEFT, padx=10)
            self.img_labels.append(lbl)

    def load_img1(self):
        path = filedialog.askopenfilename(title="Select Image 1")
        if path:
            self.img1 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.info_text.set("Image 1 loaded.")

    def load_img2(self):
        path = filedialog.askopenfilename(title="Select Image 2")
        if path:
            self.img2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.info_text.set("Image 2 loaded.")

    def show_image_on_label(self, image_cv, label, max_size=(350, 350)):
        img = cv2.cvtColor(image_cv, cv2.COLOR_GRAY2RGB)
        img = cv2.resize(img, max_size)
        img_pil = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(img_pil)
        label.configure(image=img_tk)
        label.image = img_tk

    def run_ransac(self):
        if self.img1 is None or self.img2 is None:
            messagebox.showerror("Error", "Please load both images first.")
            return

        result, info, src_pts, dst_pts, mask = align_images(self.img1, self.img2)
        if result is not None:
            vis = draw_matches(result, self.img2, src_pts, dst_pts, mask)

            # Show all images in GUI
            self.show_image_on_label(self.img1, self.img_labels[0])
            self.img_labels[0].config(text="Image 1", compound='top')

            self.show_image_on_label(self.img2, self.img_labels[1])
            self.img_labels[1].config(text="Image 2", compound='top')

            self.show_image_on_label(result, self.img_labels[2])
            self.img_labels[2].config(text="Aligned", compound='top')

            self.show_image_on_label(cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY), self.img_labels[3])
            self.img_labels[3].config(text="Matches", compound='top')

            # Update stats
            self.info_text.set(
                f"Keypoints in Image 1 : {info['keypoints_img1']}\n"
                f"Keypoints in Image 2 : {info['keypoints_img2']}\n"
                f"Total Matches         : {info['total_matches']}\n"
                f"Good Matches (0.75)   : {info['good_matches']}\n"
                f"Inliers (RANSAC)      : {info.get('inliers', 0)}\n"
                f"Inlier Ratio          : {info.get('inlier_ratio', 0.0):.2f}"
            )
        else:
            messagebox.showwarning("Warning", "Not enough good matches to align images.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RANSACApp(root)
    root.mainloop()
