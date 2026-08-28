import cv2
import numpy as np

# سازگارکننده MediaPipe برای نسخه‌های جدید (بدون solutions)
from app.services.mp_shim import face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

def get_nose_landmarks(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = image.shape
    nose_points = []
    for idx in [1, 2, 4, 5, 6, 19, 20]:  # نقاط کلیدی بینی
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        nose_points.append((x, y))
    return nose_points