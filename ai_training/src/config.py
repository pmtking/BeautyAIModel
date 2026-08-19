# src/config.py
"""
تنظیمات مرکزی پروژه BeautyAI
"""

import os
from pathlib import Path
from typing import Dict, Any

class Config:
    """کلاس تنظیمات مرکزی"""
    
    # مسیرهای پایه
    BASE_DIR = Path(__file__).parent.parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "datasets"
    LOGS_DIR = BASE_DIR / "logs"
    
    # تنظیمات مدل‌ها
    FACE_PARSER_CONFIG = {
        'min_detection_confidence': 0.5,
        'max_num_faces': 1,
        'static_image_mode': True,
        'refine_landmarks': False
    }
    
    WARPING_CONFIG = {
        'interpolation': 'linear',
        'border_mode': 'replicate',
        'default_intensity': 0.5
    }
    
    BLENDING_CONFIG = {
        'method': 'poisson',
        'fallback': 'masked',
        'blur_radius': 21
    }
    
    BEAUTY_ENGINE_CONFIG = {
        'enable_cache': True,
        'max_cache_size': 100,
        'timeout': 30
    }
    
    # ایندکس‌های نقاط صورت
    LANDMARK_INDICES = {
        'nose': list(range(1, 10)) + list(range(19, 25)),
        'lip': list(range(61, 79)) + list(range(81, 99)),
        'jaw': list(range(10, 19)) + list(range(25, 36)),
        'eye': list(range(33, 50)) + list(range(133, 150)),
        'eyebrow': list(range(55, 65)) + list(range(65, 75)),
        'face_oval': list(range(0, 17)) + list(range(17, 27))
    }
    
    @classmethod
    def get_landmark_indices(cls, area: str) -> list:
        """دریافت ایندکس‌های یک ناحیه خاص"""
        return cls.LANDMARK_INDICES.get(area, [])

# ایجاد پوشه‌های لازم
Config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
Config.DATA_DIR.mkdir(parents=True, exist_ok=True)
Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)