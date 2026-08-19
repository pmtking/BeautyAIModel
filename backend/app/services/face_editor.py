import cv2
import numpy as np
import mediapipe as mp
from .face_detector import get_nose_landmarks  # از همان تابع قبلی

def apply_minor_edit(image_bytes: bytes, edit_type: str = "nose_smaller") -> bytes:
    # تبدیل bytes به تصویر
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    
    # تشخیص نقاط بینی
    points = get_nose_landmarks(image_bytes)
    if not points:
        raise ValueError("No face detected")
    
    # محاسبه مرکز بینی
    center = np.mean(points, axis=0).astype(int)
    
    # ایجاد ماسک دایره‌ای دور بینی
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    radius = 40  # اندازه ناحیه
    cv2.circle(mask, tuple(center), radius, 255, -1)
    
    # اعمال تغییر ساده (مقیاس‌گذاری) - مثلاً کوچک‌تر کردن بینی
    if edit_type == "nose_smaller":
        scale = 0.9
    elif edit_type == "nose_bigger":
        scale = 1.1
    elif edit_type == "lips_full":
        # برای لب نیاز به نقاط دیگر داریم، فعلاً ساده می‌گیریم
        scale = 1.05
    else:
        scale = 1.0
    
    # استخراج ناحیه مورد نظر
    x, y = center
    h, w = img.shape[:2]
    
    # محدود کردن ناحیه به ابعاد تصویر
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(w, x + radius)
    y2 = min(h, y + radius)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("ROI is empty")
    
    # تغییر اندازه ناحیه (مقیاس‌گذاری)
    new_w = int(roi.shape[1] * scale)
    new_h = int(roi.shape[0] * scale)
    roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # اگر بزرگتر شد، دوباره به اندازه اصلی برگردان (برای جاگذاری)
    if scale != 1.0:
        roi_final = cv2.resize(roi_resized, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
    else:
        roi_final = roi_resized
    
    # جایگذاری در تصویر اصلی
    img[y1:y2, x1:x2] = roi_final
    
    # تبدیل به bytes برای خروجی
    _, buffer = cv2.imencode('.jpg', img)
    return buffer.tobytes()