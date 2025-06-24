# optical_flow.py

import cv2
import numpy as np
import requests

def get_video_from_url(url, output_path="temp_video.mp4"):
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
    # Video có các xe di chuyển rõ ràng
    VIDEO_URL = "https://videos.pexels.com/video-files/3845122/3845122-sd_640_360_30fps.mp4"
    video_path = get_video_from_url(VIDEO_URL)

    if video_path:
        cap = cv2.VideoCapture(video_path)

        # Tham số cho thuật toán ShiTomasi để phát hiện góc
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Tham số cho thuật toán Lucas-Kanade optical flow
        lk_params = dict(winSize=(15, 15), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        # Đọc khung hình đầu tiên và tìm các góc
        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

        # Tạo một mask để vẽ các vệt chuyển động
        mask = np.zeros_like(old_frame)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Tính toán luồng quang học
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Chọn các điểm tốt (good points)
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

                # Vẽ các vệt chuyển động
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                    frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
            
            img = cv2.add(frame, mask)
            cv2.imshow('Optical Flow - Lucas-Kanade', img)

            if cv2.waitKey(30) & 0xFF == 27: # Nhấn ESC để thoát
                break

            # Cập nhật khung hình và các điểm cho vòng lặp tiếp theo
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cv2.destroyAllWindows()
        cap.release()