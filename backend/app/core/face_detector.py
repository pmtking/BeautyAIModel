# app/core/face_detector.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import cv2
import numpy as np


class FaceDetector:
    """
    کلاس تشخیص چهره با استفاده از OpenCV
    جایگزین MediaPipe برای ویندوز
    """
    
    def __init__(self):
        # بارگذاری Haar Cascade برای تشخیص چهره
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # بارگذاری Haar Cascade برای تشخیص چشم
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # بارگذاری Haar Cascade برای تشخیص لبخند
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

    def detect_landmarks(self, image_path: str):
        """
        تشخیص چهره و استخراج نقاط کلیدی با OpenCV
        
        Args:
            image_path: مسیر تصویر
            
        Returns:
            list: لیست نقاط کلیدی به صورت {'x': float, 'y': float, 'z': float}
                  یا None در صورت عدم تشخیص چهره
        """
        # خواندن تصویر
        image = cv2.imread(image_path)
        if image is None:
            return None

        # تبدیل به خاکستری
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # تشخیص چهره
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # اگر چهره‌ای تشخیص داده نشد
        if len(faces) == 0:
            return None
        
        # گرفتن اولین چهره
        x, y, w, h = faces[0]
        h_img, w_img = image.shape[:2]
        
        # تشخیص چشم‌ها برای دقت بیشتر
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray)
        
        # ایجاد نقاط کلیدی
        landmarks = self._create_landmarks(x, y, w, h, w_img, h_img, eyes)
        
        return landmarks

    def _create_landmarks(self, x, y, w, h, w_img, h_img, eyes):
        """
        ایجاد نقاط کلیدی شبیه‌سازی‌شده برای تست
        
        Args:
            x, y, w, h: موقعیت چهره
            w_img, h_img: ابعاد تصویر
            eyes: موقعیت چشم‌ها
            
        Returns:
            list: نقاط کلیدی
        """
        landmarks = []
        
        # ============================================
        # 1. نقاط بینی (10 نقطه)
        # ============================================
        nose_center = (x + w//2, y + int(h * 0.6))
        
        # مرکز بینی
        landmarks.append({
            'x': round(nose_center[0] / w_img * 100, 2),
            'y': round(nose_center[1] / h_img * 100, 2),
            'z': 0.0
        })
        
        # نقاط اطراف بینی (حلقه)
        for i in range(1, 10):
            angle = (i / 10) * 2 * np.pi
            radius = int(w * 0.08)
            px = nose_center[0] + int(radius * np.cos(angle))
            py = nose_center[1] + int(radius * np.sin(angle) * 0.7)
            landmarks.append({
                'x': round(px / w_img * 100, 2),
                'y': round(py / h_img * 100, 2),
                'z': 0.0
            })
        
        # ============================================
        # 2. نقاط لب (10 نقطه)
        # ============================================
        lip_center = (x + w//2, y + int(h * 0.78))
        
        for i in range(10, 20):
            angle = ((i - 10) / 10) * 2 * np.pi
            radius = int(w * 0.12)
            px = lip_center[0] + int(radius * np.cos(angle))
            py = lip_center[1] + int(radius * np.sin(angle) * 0.4)
            landmarks.append({
                'x': round(px / w_img * 100, 2),
                'y': round(py / h_img * 100, 2),
                'z': 0.0
            })
        
        # ============================================
        # 3. نقاط فک (12 نقطه)
        # ============================================
        for i in range(20, 32):
            angle = ((i - 20) / 12) * np.pi + np.pi/6
            radius = int(w * 0.35)
            px = x + w//2 + int(radius * np.cos(angle))
            py = y + int(h * 0.5) + int(radius * np.sin(angle) * 0.7)
            landmarks.append({
                'x': round(px / w_img * 100, 2),
                'y': round(py / h_img * 100, 2),
                'z': 0.0
            })
        
        # ============================================
        # 4. نقاط چشم‌ها (16 نقطه)
        # ============================================
        # تشخیص موقعیت چشم‌ها
        if len(eyes) >= 2:
            eye_positions = []
            for (ex, ey, ew, eh) in eyes[:2]:
                eye_positions.append((x + ex + ew//2, y + ey + eh//2))
        else:
            # موقعیت پیش‌فرض
            eye_positions = [
                (x + int(w * 0.28), y + int(h * 0.35)),
                (x + int(w * 0.72), y + int(h * 0.35))
            ]
        
        # نقاط اطراف هر چشم
        for idx, (eye_x, eye_y) in enumerate(eye_positions):
            start_idx = 32 + (idx * 8)
            for i in range(start_idx, start_idx + 8):
                angle = ((i - start_idx) / 8) * 2 * np.pi
                radius = int(w * 0.06)
                px = eye_x + int(radius * np.cos(angle))
                py = eye_y + int(radius * np.sin(angle) * 0.6)
                landmarks.append({
                    'x': round(px / w_img * 100, 2),
                    'y': round(py / h_img * 100, 2),
                    'z': 0.0
                })
        
        return landmarks

    def get_indices(self) -> dict:
        """
        ایندکس‌های مهم نقاط صورت برای تغییرات
        
        Returns:
            dict: شامل ایندکس‌های بینی، لب، فک و چشم
        """
        return {
            'nose': list(range(0, 10)),      # 0-9
            'lip': list(range(10, 20)),      # 10-19
            'jaw': list(range(20, 32)),      # 20-31
            'eye': list(range(32, 48))       # 32-47
        }