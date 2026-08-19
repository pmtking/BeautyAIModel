# app/services/face_service.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import cv2
import numpy as np
import tempfile
import os
import sys
from pathlib import Path
from app.core.face_detector import FaceDetector
from app.utils.warping import FaceWarping
from app.utils.blending import FaceBlending

# اضافه کردن مسیر مدل
MODEL_PATH = Path(__file__).parent.parent.parent.parent / 'ai_training/src/models'
sys.path.insert(0, str(MODEL_PATH))


class FaceService:
    """سرویس مدیریت تشخیص و تغییر شکل صورت"""
    
    def __init__(self):
        self.detector = FaceDetector()
        self.warping = FaceWarping()
        self.blending = FaceBlending()
        self.ai_engine = None
        self._load_ai_engine()
    
    def _load_ai_engine(self):
        """بارگذاری موتور AI"""
        try:
            from beauty_engine.model import BeautyEngineModel
            self.ai_engine = BeautyEngineModel()
            print("✅ AI Engine loaded")
        except Exception as e:
            print(f"⚠️ AI Engine not available: {e}")
            self.ai_engine = None
    
    def process_with_ai(self, image_bytes: bytes, text: str) -> dict:
        """
        پردازش با موتور AI
        
        Args:
            image_bytes: بایت‌های تصویر
            text: درخواست کاربر (متن فارسی)
            
        Returns:
            dict: نتیجه پردازش
        """
        if self.ai_engine is None:
            return {
                'status': 'error',
                'message': 'موتور AI در دسترس نیست'
            }
        
        try:
            # تبدیل به تصویر
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {'status': 'error', 'message': 'تصویر معتبر نیست'}
            
            # پردازش با AI
            result = self.ai_engine.process(image, text)
            
            if result['status'] == 'error':
                return result
            
            # تبدیل به بایت
            _, buffer = cv2.imencode('.jpg', result['image'])
            
            return {
                'status': 'success',
                'image_bytes': buffer.tobytes(),
                'description': result.get('description', ''),
                'changes': result.get('changes', {})
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'خطا در پردازش AI: {str(e)}'
            }
    
    def analyze_image(self, image_bytes: bytes) -> dict:
        # ... کدهای موجود ...
        pass
    
    def edit_image(self, image_bytes: bytes, feature: str, intensity: float) -> dict:
        # ... کدهای موجود ...
        pass