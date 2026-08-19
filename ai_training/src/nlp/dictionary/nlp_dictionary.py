
import re
import cv2
import numpy as np
from typing import Dict, Optional, List
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

from face_parser.model import FaceParserModel
from warping.model import WarpingModel
from blending.model import BlendingModel

# ============================================
# ✅ اضافه کردن مسیر دیکشنری NLP
# ============================================
NLP_PATH = Path(__file__).parent.parent.parent / 'nlp'
sys.path.insert(0, str(NLP_PATH))

try:
    from dictionary.nlp_dictionary import AREA_KEYWORDS, ACTION_KEYWORDS, STYLE_KEYWORDS
    logger.info("✅ NLP Dictionary keywords loaded")
except ImportError as e:
    logger.warning(f"⚠️ NLP Dictionary not loaded: {e}")
    AREA_KEYWORDS = {}
    ACTION_KEYWORDS = {}
    STYLE_KEYWORDS = {}


class BeautyEngineModel:
    def __init__(self):
        self.face_parser = FaceParserModel()
        self.warping = WarpingModel()
        self.blending = BlendingModel()
        self.unit_values = {'cc': 0.6, 'ml': 0.6, 'syringe': 0.8}
        self.PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
        
        # ============================================
        # دیکشنری‌های داخلی (اگر دیکشنری خارجی کار نکرد)
        # ============================================
        self.AREA_KEYWORDS = AREA_KEYWORDS or {
            'nose': ['بینی', 'دماغ', 'دماغم', 'بینی من', 'دماغ من', 'نوک بینی', 'پل بینی'],
            'lip': ['لب', 'لبام', 'لبامو', 'لب من', 'دهان', 'لب بالا', 'لب پایین', 'گوشه لب'],
            'jaw': ['فک', 'چانه', 'فکم', 'چانم', 'فک من', 'خط فک'],
            'cheek': ['گونه', 'گونه‌ها', 'گونه من', 'استخوان گونه'],
            'forehead': ['پیشانی', 'پیشونم', 'پیشانی من'],
            'eye': ['چشم', 'چشمام', 'چشم من', 'چشم‌ها', 'پلک']
        }
        
        self.ACTION_KEYWORDS = ACTION_KEYWORDS or {
            'smaller': ['کوچیک‌تر', 'کوچک‌تر', 'باریک‌تر', 'کم‌تر', 'کم', 'نازک‌تر', 'کاهش', 'ریزتر'],
            'bigger': ['بزرگ‌تر', 'بلندتر', 'بیشتر', 'بزرگ', 'درشت‌تر', 'پهن‌تر', 'افزایش'],
            'fuller': ['پرتر', 'حجم‌تر', 'پرشد', 'برجسته‌تر', 'حجیم‌تر', 'پر'],
            'sharper': ['تیزتر', 'مشخص‌تر', 'زاویه‌دارتر'],
            'smoother': ['صاف‌تر', 'یکدست‌تر', 'نرم‌تر'],
            'lift': ['لیفت', 'بالا', 'بالا بردن', 'کشیدن بالا']
        }
        
        self.STYLE_KEYWORDS = STYLE_KEYWORDS or {
            'lip': {'قلوه‌ای': 'heart_shape', 'روسی': 'russian', 'طبیعی': 'natural'},
            'nose': {'قلمی': 'slim_bridge', 'عروسکی': 'doll_tip', 'طبیعی': 'natural'},
        }
        
        logger.info("✅ BeautyEngineModel ready")

    def process(self, image: np.ndarray, text: str, intensity: Optional[float] = None) -> Dict:
        landmarks = self.face_parser.detect_from_image(image)
        if not landmarks:
            return {'status': 'error', 'message': 'چهره‌ای شناسایی نشد'}

        parsed = self._parse_request(text)
        if not parsed.get('area') or not parsed.get('action'):
            return {'status': 'error', 'message': f'متوجه نشدم: "{text}"'}

        points = self.face_parser.get_points(landmarks, parsed['area'])
        if not points:
            return {'status': 'error', 'message': f'ناحیه {parsed["area"]} پیدا نشد'}

        intensity = intensity or parsed.get('intensity', 0.5)
        warped = self.warping.warp(image, points, intensity, parsed['action'], parsed['area'])
        mask = self.warping.create_mask(image, points)
        result = self.blending.blend(image, warped, mask)

        return {
            'status': 'success',
            'image': result,
            'original': image,
            'changes': parsed,
            'intensity': intensity,
            'description': self._generate_description(parsed, intensity),
            'message': '✅ تغییرات اعمال شد'
        }

    def _parse_request(self, text: str) -> Dict:
        """تحلیل متن با دیکشنری‌های کامل"""
        text_lower = text.lower()
        result = {
            'area': None,
            'action': None,
            'style': None,
            'intensity': 0.5,
            'confidence': 0.0
        }

        # ============================================
        # ۱. تشخیص ناحیه (Area)
        # ============================================
        for area, keywords in self.AREA_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    result['area'] = area
                    result['confidence'] += 0.3
                    break
            if result['area']:
                break

        # ============================================
        # ۲. تشخیص استایل (Style) - اولویت اول
        # ============================================
        if result['area'] and result['area'] in self.STYLE_KEYWORDS:
            styles = self.STYLE_KEYWORDS[result['area']]
            for kw, action in styles.items():
                if kw in text_lower:
                    result['action'] = action
                    result['style'] = action
                    result['confidence'] += 0.3
                    break

        # ============================================
        # ۳. تشخیص عمل (Action) - اگر استایل نبود
        # ============================================
        if not result['action']:
            for action, keywords in self.ACTION_KEYWORDS.items():
                for kw in keywords:
                    if kw in text_lower:
                        result['action'] = action
                        result['confidence'] += 0.3
                        break
                if result['action']:
                    break

        # ============================================
        # ۴. تشخیص شدت (Intensity)
        # ============================================
        if 'کم' in text_lower or 'ملایم' in text_lower:
            result['intensity'] = 0.3
        elif 'زیاد' in text_lower or 'خیلی' in text_lower:
            result['intensity'] = 0.8
        elif 'متوسط' in text_lower:
            result['intensity'] = 0.5

        # ============================================
        # ۵. تشخیص مقدار عددی (cc/ml)
        # ============================================
        match = re.search(r'(\d+)\s*(cc|سیسی|ml)', text.translate(self.PERSIAN_DIGITS), re.I)
        if match:
            value = float(match.group(1))
            result['intensity'] = min(value * 0.6, 1.0)
            result['amount'] = {'value': value, 'unit': 'cc'}

        return result

    def _generate_description(self, parsed: Dict, intensity: float) -> str:
        area_names = {
            'nose': 'بینی', 'lip': 'لب', 'jaw': 'فک',
            'cheek': 'گونه', 'forehead': 'پیشانی', 'eye': 'چشم'
        }
        action_names = {
            'smaller': 'کوچک‌تر', 'bigger': 'بزرگ‌تر',
            'fuller': 'پرتر', 'sharper': 'تیزتر',
            'smoother': 'صاف‌تر', 'lift': 'لیفت',
            'heart_shape': 'قلوه‌ای', 'slim_bridge': 'قلمی',
            'doll_tip': 'عروسکی', 'russian': 'روسی'
        }
        area = area_names.get(parsed.get('area', ''), 'ناحیه')
        action = action_names.get(parsed.get('action', ''), 'تغییر')

        if parsed.get('amount'):
            amount = parsed['amount']
            return f"{area} با {amount['value']} سی‌سی {action} می‌شود"

        return f"{area} با شدت {int(intensity*100)}% {action} می‌شود"


beauty_engine = BeautyEngineModel()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing BeautyEngine")
    print("=" * 60)
    
    test_texts = [
        "دماغم رو کوچیکتر کن",
        "لبامو قلوه‌ای کن با ۲ سی‌سی",
        "فکم رو تیز کن",
        "بینی منو قلمی کن",
    ]
    
    for text in test_texts:
        result = beauty_engine._parse_request(text)
        print(f"\n📝 '{text}'")
        print(f"   Area: {result.get('area')}")
        print(f"   Action: {result.get('action')}")
        print(f"   Intensity: {result.get('intensity')}")
        print(f"   Confidence: {result.get('confidence', 0):.2f}")
