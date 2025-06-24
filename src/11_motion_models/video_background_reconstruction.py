# video_background_reconstruction.py

import cv2
import numpy as np
import requests

def get_video_from_url(url, output_path="temp_video_bg.mp4"):
    """Tải video từ URL và lưu tạm thời."""
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return output_path
    except Exception as e:
        print(f"Lỗi tải video: {e}")
        return None

if __name__ == "__main__":
    VIDEO_URL = "https://videos.pexels.com/video-files/855321/855321-sd_640_360_30fps.mp4" # Video người đi bộ
    video_path = get_video_from_url(VIDEO_URL)

    if video_path:
        cap = cv2.VideoCapture(video_path)
        
        # Đọc một vài khung hình đầu tiên để xử lý
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frames_to_process = min(frame_count, 100) # Xử lý tối đa 100 frame để demo nhanh
        
        print(f"Sẽ xử lý {num_frames_to_process} khung hình...")

        # Đọc khung hình đầu tiên làm mốc (anchor frame)
        _, first_frame = cap.read()
        h, w = first_frame.shape[:2]

        # Khởi tạo ORB detector
        orb = cv2.ORB_create()
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Lưu trữ các frame đã được căn chỉnh
        aligned_frames = [first_frame]

        for i in range(1, num_frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            print(f"Đang căn chỉnh khung hình {i+1}/{num_frames_to_process}...")
            
            # Căn chỉnh frame hiện tại với first_frame
            kp1, des1 = orb.detectAndCompute(first_frame, None)
            kp2, des2 = orb.detectAndCompute(frame, None)
            
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                if len(matches) > 10:
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    
                    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    if M is not None:
                        # Warp frame hiện tại để nó thẳng hàng với frame đầu tiên
                        warped_frame = cv2.warpPerspective(frame, M, (w, h))
                        aligned_frames.append(warped_frame)

        cap.release()

        if len(aligned_frames) > 1:
            print("Đang tính toán ảnh nền từ các khung hình đã căn chỉnh...")
            # Xếp chồng các ảnh đã căn chỉnh
            stacked_frames = np.stack(aligned_frames, axis=0)
            
            # Tính giá trị trung vị (median) dọc theo trục thời gian
            # Median rất mạnh trong việc loại bỏ các outliers (đối tượng chuyển động)
            background = np.median(stacked_frames, axis=0).astype(np.uint8)

            cv2.imshow("Reconstructed Background", background)
            cv2.imshow("Original First Frame", first_frame)
            print("Đã xong! So sánh ảnh nền tái tạo và ảnh gốc.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()