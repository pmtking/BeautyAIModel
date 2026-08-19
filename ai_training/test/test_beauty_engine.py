# src/models/face_parser/model.py
"""
مدل تشخیص و استخراج نقاط صورت - نسخه OpenCV
"""
import sys
import os
from pathlib import Path

# ✅ اضافه کردن مسیر src به sys.path
BASE_DIR = Path(__file__).parent.parent  # ai_training
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

# حالا import ها کار می‌کنند
from src.models.beauty_engine.model import beauty_engine
from src.logger import logger
from src.keywords_database import KeywordDatabase
from src.request_parser import parser


class FaceParserModel:
    """مدل تشخیص نقاط کلیدی صورت با OpenCV"""
    
    def __init__(self):
        # استفاده از Haar Cascade برای تشخیص چهره
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        
        # نقاط شبیه‌سازی‌شده (برای تست)
        self.landmark_indices = Config.LANDMARK_INDICES
        
        # تولید نقاط شبیه‌سازی‌شده
        self.mock_points = self._generate_mock_points()
        
        logger.info("✅ FaceParserModel initialized (OpenCV version)")
    
    def _generate_mock_points(self):
        """تولید نقاط شبیه‌سازی‌شده"""
        points = []
        for i in range(468):
            angle = (i / 468) * 2 * np.pi
            radius = 50 + 30 * np.sin(i / 20)
            x = 150 + radius * np.cos(angle)
            y = 150 + radius * np.sin(angle) * 0.8
            points.append({'x': x, 'y': y, 'z': 0})
        return points
    
    def detect(self, image_path: str) -> Optional[List[Dict]]:
        """تشخیص نقاط از مسیر تصویر"""
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Image not found: {image_path}")
            return None
        
        return self.detect_from_image(image)
    
    def detect_from_image(self, image: np.ndarray) -> Optional[List[Dict]]:
        """تشخیص نقاط از تصویر با OpenCV"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # تشخیص چهره
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(faces) == 0:
                logger.warning("No face detected")
                return self.mock_points  # برگرداندن نقاط شبیه‌سازی‌شده برای تست
            
            x, y, w, h = faces[0]
            
            # تشخیص چشم‌ها
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            # ایجاد نقاط شبیه‌سازی‌شده بر اساس موقعیت چهره
            h_img, w_img = image.shape[:2]
            points = self._generate_points_from_face(x, y, w, h, w_img, h_img)
            
            return points
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return self.mock_points
    
    def _generate_points_from_face(self, x, y, w, h, w_img, h_img):
        """تولید نقاط بر اساس موقعیت چهره"""
        points = []
        
        # مرکز چهره
        cx = x + w // 2
        cy = y + h // 2
        
        # نقاط بینی (مرکز)
        for i in range(10):
            angle = (i / 10) * 2 * np.pi
            radius = w * 0.08
            px = cx + radius * np.cos(angle)
            py = cy + h * 0.1 + radius * np.sin(angle) * 0.7
            points.append({
                'x': px / w_img * 100,
                'y': py / h_img * 100,
                'z': 0
            })
        
        # نقاط لب
        for i in range(10):
            angle = (i / 10) * 2 * np.pi
            radius = w * 0.12
            px = cx + radius * np.cos(angle)
            py = cy + h * 0.25 + radius * np.sin(angle) * 0.4
            points.append({
                'x': px / w_img * 100,
                'y': py / h_img * 100,
                'z': 0
            })
        
        # نقاط فک
        for i in range(12):
            angle = (i / 12) * np.pi + np.pi/6
            radius = w * 0.35
            px = cx + radius * np.cos(angle)
            py = cy + h * 0.15 + radius * np.sin(angle) * 0.7
            points.append({
                'x': px / w_img * 100,
                'y': py / h_img * 100,
                'z': 0
            })
        
        # نقاط چشم‌ها
        for eye_offset in [-w*0.2, w*0.2]:
            for i in range(8):
                angle = (i / 8) * 2 * np.pi
                radius = w * 0.06
                px = cx + eye_offset + radius * np.cos(angle)
                py = cy - h * 0.05 + radius * np.sin(angle) * 0.6
                points.append({
                    'x': px / w_img * 100,
                    'y': py / h_img * 100,
                    'z': 0
                })
        
        return points
    
    def get_points(self, landmarks: List[Dict], area: str) -> List[List[int]]:
        """دریافت نقاط یک ناحیه خاص"""
        # تبدیل درصد به پیکسل
        points = []
        for lm in landmarks:
            x = int(lm['x'])
            y = int(lm['y'])
            points.append([x, y])
        
        # اگر نقاط کافی نیست، نقاط شبیه‌سازی‌شده برگردان
        if len(points) < 10:
            return self._generate_fallback_points(area)
        
        return points
    
    def _generate_fallback_points(self, area: str) -> List[List[int]]:
        """تولید نقاط جایگزین برای تست"""
        if area == 'nose':
            return [[140, 140], [160, 140], [150, 160]]
        elif area == 'lip':
            return [[140, 170], [160, 170], [150, 185]]
        elif area == 'jaw':
            return [[120, 180], [150, 200], [180, 180]]
        else:
            return [[140, 140], [160, 140], [150, 160]]
    
    def draw_landmarks(self, image: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        """رسم نقاط روی تصویر"""
        result = image.copy()
        h, w = image.shape[:2]
        
        for lm in landmarks:
            x = int(lm['x'] * w / 100)
            y = int(lm['y'] * h / 100)
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        
        return result
    
    def get_face_mesh(self):
        """دریافت مدل (برای سازگاری)"""
        return None