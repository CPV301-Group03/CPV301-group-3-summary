import os
import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, DISABLED, NORMAL
from PIL import Image, ImageTk

DATASET_PATH = "images"
IMG_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 4000

def preprocess_face(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def read_faces_and_labels(dataset_path):
    faces, labels, label_map = [], [], {}
    current_label = 0
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    for folder in sorted(os.listdir(dataset_path)):
        person_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(person_path):
            continue
        label_map[current_label] = folder
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img0 = cv2.imread(img_path)
            if img0 is None:
                continue
            gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
            faces_rects = face_cascade.detectMultiScale(gray, 1.2, 5)
            if len(faces_rects) > 0:
                (x, y, w, h) = faces_rects[0]  # chỉ lấy khuôn mặt đầu tiên
                face = gray[y:y+h, x:x+w]
                face_processed = preprocess_face(face)
                faces.append(face_processed)
                labels.append(current_label)
        current_label += 1
    return faces, labels, label_map

def train_model():
    faces, labels, label_map = read_faces_and_labels(DATASET_PATH)
    recognizer = cv2.face.EigenFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer, label_map

# Biến lưu ảnh vừa chọn
current_img_path = None

def choose_image():
    global current_img_path
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png")])
    if file_path:
        current_img_path = file_path
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = pil_img.resize((300, 300))
        tk_img = ImageTk.PhotoImage(pil_img)
        image_label.config(image=tk_img)
        image_label.image = tk_img
        result_label.config(text="")  # clear kết quả cũ
        recog_btn.config(state=NORMAL)
    else:
        recog_btn.config(state=DISABLED)

def recognize_face():
    global current_img_path
    if not current_img_path:
        result_label.config(text="Chưa chọn ảnh!")
        return
    image = cv2.imread(current_img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    if len(faces) == 0:
        result_label.config(text="Không phát hiện khuôn mặt.")
        return
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        face_processed = preprocess_face(roi)
        label, confidence = recognizer.predict(face_processed)
        if confidence > CONFIDENCE_THRESHOLD:
            name = "Unknown"
        else:
            name = label_map.get(label, "Unknown")
        result_text = f"Nhận diện: {name}"
        result_label.config(text=result_text)
        # Vẽ khung lên ảnh và cập nhật lại khung hiển thị
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img = pil_img.resize((300, 300))
    tk_img = ImageTk.PhotoImage(pil_img)
    image_label.config(image=tk_img)
    image_label.image = tk_img

# ---------------- GUI ----------------
recognizer, label_map = train_model()
app = Tk()
app.title("Face Recognition Workshop (GUI ready)")
app.geometry("420x480")
Button(app, text="Choose Image", command=choose_image, font=("Arial", 12)).pack(pady=10)
recog_btn = Button(app, text="Recognition", command=recognize_face, font=("Arial", 12), state=DISABLED)
recog_btn.pack(pady=5)
image_label = Label(app)
image_label.pack(pady=10)
result_label = Label(app, text="", font=("Arial", 13))
result_label.pack(pady=10)
app.mainloop()

