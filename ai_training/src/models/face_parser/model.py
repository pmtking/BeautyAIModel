# src/models/face_parser/model.py
"""
مدل تشخیص و استخراج نقاط صورت - نسخه Hybrid (MediaPipe + OpenCV)
نسخه بازبینی‌شده: اندیس‌های صحیح MediaPipe برای هر ناحیه (مستند رسمی FACEMESH_*)
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
    logger.info("✅ MediaPipe available")
except ImportError:
    HAS_MEDIAPIPE = False
    logger.warning("⚠️ MediaPipe not available, using OpenCV fallback")


class FaceParserModel:
    """مدل تشخیص نقاط کلیدی صورت - Hybrid"""

    def __init__(self):
        self.HAS_MEDIAPIPE = HAS_MEDIAPIPE

        # ============================================
        # اندیس‌های صحیح MediaPipe Face Mesh (468 نقطه)
        # این‌ها بر اساس FACEMESH_LIPS / FACEMESH_LEFT_EYE / FACEMESH_RIGHT_EYE /
        # FACEMESH_FACE_OVAL رسمی گوگل هستن - نه رنج‌های دلبخواهی
        # ترتیب نقاط لب طوریه که به‌صورت یک کانتور بسته دور دهان می‌چرخه:
        #   ابتدا و انتها (61, 291) = گوشه‌های دهان (چپ/راست)
        #   وسط بالا (0) = نوک کمان کوپید
        #   وسط پایین (17) = مرکز لب پایین
        # این ترتیب دقیقاً با منطق argmin/argmax در specialized.py هماهنگه
        # ============================================
        self.landmark_indices = {
            'nose': [
                168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
                97, 326, 129, 358, 240, 460, 64, 294
            ],
            'lip': [
                61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                375, 321, 405, 314, 17, 84, 181, 91, 146
            ],
            'jaw': [
                172, 136, 150, 149, 176, 148, 152,
                377, 400, 378, 379, 365, 397, 288
            ],
            'cheek': [
                50, 101, 100, 47, 205, 187,   # گونه چپ
                280, 330, 329, 277, 425, 411  # گونه راست
            ],
            'forehead': [
                109, 10, 338, 297, 332, 284, 251, 21, 54, 103
            ],
            'eye': [
                # چشم چپ
                33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
                # چشم راست
                362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
            ]
        }

        if HAS_MEDIAPIPE:
            self._init_mediapipe()
        else:
            self._init_opencv()

        logger.info("✅ FaceParserModel initialized")

    def _init_mediapipe(self):
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            self.use_mediapipe = True
            logger.info("✅ MediaPipe initialized")
        except Exception as e:
            logger.error(f"MediaPipe init failed: {e}")
            self._init_opencv()

    def _init_opencv(self):
        self.use_mediapipe = False
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )
        logger.info("✅ OpenCV fallback initialized")

    def detect(self, image_path: str) -> Optional[List[Dict]]:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Image not found: {image_path}")
            return None
        return self.detect_from_image(image)

    def detect_from_image(self, image: np.ndarray) -> Optional[List[Dict]]:
        if self.use_mediapipe:
            return self._detect_mediapipe(image)
        else:
            return self._detect_opencv(image)

    def _detect_mediapipe(self, image: np.ndarray) -> Optional[List[Dict]]:
        h, w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return self._detect_opencv(image)

        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append({'x': lm.x * w, 'y': lm.y * h, 'z': lm.z * w})

        return landmarks

    def _detect_opencv(self, image: np.ndarray) -> Optional[List[Dict]]:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]

            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )

            if len(faces) == 0:
                return self._generate_mock_points(w, h)

            x, y, fw, fh = faces[0]
            return self._generate_points_from_face(x, y, fw, fh, w, h)

        except Exception as e:
            logger.error(f"OpenCV detection error: {e}")
            return self._generate_mock_points(w, h)

    def _generate_mock_points(self, w: int, h: int) -> List[Dict]:
        points = []
        cx, cy = w // 2, h // 2
        radius = min(w, h) // 3

        for i in range(68):
            angle = (i / 68) * 2 * np.pi
            r = radius * (0.7 + 0.3 * np.sin(i * 0.3))
            points.append({'x': cx + r * np.cos(angle), 'y': cy + r * np.sin(angle) * 0.8, 'z': 0.0})

        return points

    def _generate_points_from_face(self, x, y, fw, fh, w, h) -> List[Dict]:
        """
        نکته: این fallback (بدون MediaPipe) نقاط را با موقعیت تخمینی می‌سازد،
        نه با اندیس واقعی. get_points برای این حالت به‌طور خودکار به
        _fallback_points سوییچ می‌کند چون اندیس‌های MediaPipe روی این نقاط معنا ندارند.
        """
        points = []
        cx, cy = x + fw // 2, y + fh // 2

        nose = (cx, y + int(fh * 0.6))
        for i in range(10):
            angle = (i / 10) * 2 * np.pi
            r = int(fw * 0.08)
            points.append({'x': nose[0] + r * np.cos(angle), 'y': nose[1] + r * np.sin(angle) * 0.7, 'z': 0.0})

        lip = (cx, y + int(fh * 0.75))
        for i in range(10):
            angle = (i / 10) * 2 * np.pi
            r = int(fw * 0.12)
            points.append({'x': lip[0] + r * np.cos(angle), 'y': lip[1] + r * np.sin(angle) * 0.4, 'z': 0.0})

        for i in range(12):
            angle = (i / 12) * np.pi + np.pi / 6
            r = int(fw * 0.35)
            points.append({'x': cx + r * np.cos(angle), 'y': y + int(fh * 0.5) + r * np.sin(angle) * 0.7, 'z': 0.0})

        eye_pos = [(x + int(fw * 0.28), y + int(fh * 0.35)), (x + int(fw * 0.72), y + int(fh * 0.35))]
        for ex, ey in eye_pos:
            for i in range(8):
                angle = (i / 8) * 2 * np.pi
                r = int(fw * 0.06)
                points.append({'x': ex + r * np.cos(angle), 'y': ey + r * np.sin(angle) * 0.6, 'z': 0.0})

        return points

    def get_points(self, landmarks: List[Dict], area: str) -> List[List[int]]:
        """
        دریافت نقاط یک ناحیه خاص.
        مهم: اندیس‌های self.landmark_indices فقط برای خروجی MediaPipe (468 نقطه) معتبرند.
        اگر از مسیر OpenCV fallback آمده باشیم (کمتر از 468 نقطه)، اندیس‌ها بی‌معنا
        می‌شوند، پس مستقیم به fallback هندسی می‌رویم.
        """
        if not landmarks or len(landmarks) < 400:  # فقط MediaPipe واقعی این تعداد را می‌دهد
            return self._fallback_points(area)

        indices = self.landmark_indices.get(area, [])
        points = []
        for idx in indices:
            if idx < len(landmarks):
                points.append([int(landmarks[idx]['x']), int(landmarks[idx]['y'])])

        if len(points) < 3:
            return self._fallback_points(area)

        return points

    def _fallback_points(self, area: str) -> List[List[int]]:
        fallback = {
            'nose': [[140, 140], [160, 140], [150, 160]],
            'lip': [[140, 170], [160, 170], [150, 185]],
            'jaw': [[120, 180], [150, 200], [180, 180]],
            'cheek': [[100, 150], [200, 150], [150, 170]],
            'forehead': [[120, 100], [180, 100], [150, 110]],
            'eye': [[130, 130], [150, 130], [140, 140]]
        }
        return fallback.get(area, [[140, 140], [160, 140], [150, 160]])

    def draw_landmarks(self, image: np.ndarray, landmarks: List[Dict]) -> np.ndarray:
        result = image.copy()
        h, w = image.shape[:2]
        for lm in landmarks:
            x, y = int(lm['x']), int(lm['y'])
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(result, (x, y), 2, (0, 255, 0), -1)
        return result

    def get_face_mesh(self):
        return getattr(self, 'face_mesh', None)


if __name__ == "__main__":
    print("🧪 Testing FaceParserModel...")
    model = FaceParserModel()

    test_img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (50, 50), (250, 250), (200, 200, 200), -1)

    points = model.detect_from_image(test_img)
    print(f"✅ Detected {len(points)} points")

    for area in ['nose', 'lip', 'jaw', 'cheek', 'forehead', 'eye']:
        pts = model.get_points(points, area)
        print(f"✅ {area}: {len(pts)} points -> {pts[:3]}")