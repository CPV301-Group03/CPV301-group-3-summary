import cv2
import numpy as np

def nothing(x):
    pass

# Load Haar cascade để phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tạo cửa sổ GUI
cv2.namedWindow('Face + HOG GUI')

# Trackbars điều chỉnh tham số HOG
cv2.createTrackbar('blockSize', 'Face + HOG GUI', 16, 64, nothing)
cv2.createTrackbar('cellSize', 'Face + HOG GUI', 8, 32, nothing)
cv2.createTrackbar('nbins', 'Face + HOG GUI', 9, 18, nothing)
cv2.createTrackbar('winSigma x10', 'Face + HOG GUI', 0, 50, nothing)  # dùng *0.1 để tạo float

# Mở webcam
cap = cv2.VideoCapture(0)
print("Đang chạy... Nhấn 'q' để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize nhỏ để đỡ lag
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Đọc thông số từ trackbar
    block = max(8, cv2.getTrackbarPos('blockSize', 'Face + HOG GUI') // 8 * 8)
    cell = max(4, cv2.getTrackbarPos('cellSize', 'Face + HOG GUI') // 4 * 4)
    nbins = max(1, cv2.getTrackbarPos('nbins', 'Face + HOG GUI'))
    win_sigma = cv2.getTrackbarPos('winSigma x10', 'Face + HOG GUI') / 10.0
    win_sigma = win_sigma if win_sigma > 0 else -1.0  # -1.0 = tự động

    # Lặp qua từng khuôn mặt detect được
    for (x, y, w, h) in faces:
        # Vẽ khung
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Cắt ảnh khuôn mặt và resize về đúng winSize
        face_img = gray[y:y+h, x:x+w]
        resized_face = cv2.resize(face_img, (64, 128))

        try:
            # Khởi tạo HOG descriptor
            hog = cv2.HOGDescriptor(
                _winSize=(64, 128),
                _blockSize=(block, block),
                _blockStride=(cell, cell),
                _cellSize=(cell, cell),
                _nbins=nbins,
                _derivAperture=1,
                _winSigma=win_sigma,
                _histogramNormType=0,
                _L2HysThreshold=0.2,
                _gammaCorrection=True,
                _nlevels=64,
                _signedGradient=False
            )

            # Tính HOG vector đặc trưng (feature)
            features = hog.compute(resized_face)
            cv2.putText(frame, f"HOG: {len(features)} dims", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        except Exception as e:
            print(f"Lỗi HOG config: {e}")

    # Hiển thị kết quả
    cv2.imshow('Face + HOG GUI', frame)

    # Thoát bằng phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
