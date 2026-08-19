import re
import cv2
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

from face_parser.model import FaceParserModel
from warping.model import WarpingModel
from blending.model import BlendingModel


class BeautyEngineModel:
    def __init__(self):
        self.face_parser = FaceParserModel()
        self.warping = WarpingModel()
        self.blending = BlendingModel()
        self.unit_values = {'cc': 0.6, 'ml': 0.6, 'syringe': 0.8}
        self.PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
        
        self.AREA_KEYWORDS = {
            'nose': ['بینی', 'دماغ', 'دماغم', 'بینی من', 'دماغ من', 'نوک بینی', 'پل بینی'],
            'lip': ['لب', 'لبام', 'لبامو', 'لب من', 'دهان', 'لب بالا', 'لب پایین', 'گوشه لب'],
            'jaw': ['فک', 'چانه', 'فکم', 'چانم', 'فک من', 'خط فک'],
            'cheek': ['گونه', 'گونه‌ها', 'گونه من', 'استخوان گونه'],
            'forehead': ['پیشانی', 'پیشونم', 'پیشانی من'],
            'eye': ['چشم', 'چشمام', 'چشم من', 'چشم‌ها', 'پلک']
        }
        
        self.ACTION_KEYWORDS = {
            'smaller': ['کوچیک‌تر', 'کوچک‌تر', 'کوچیک', 'کوچک', 'باریک‌تر', 'کم‌تر', 'کم', 'نازک‌تر', 'کاهش', 'ریزتر'],
            'bigger': ['بزرگ‌تر', 'بلندتر', 'بیشتر', 'بزرگ', 'درشت‌تر', 'پهن‌تر', 'افزایش'],
            'fuller': ['پرتر', 'حجم‌تر', 'پرشد', 'برجسته‌تر', 'حجیم‌تر', 'پر'],
            'sharper': ['تیزتر', 'مشخص‌تر', 'زاویه‌دارتر'],
            'smoother': ['صاف‌تر', 'یکدست‌تر', 'نرم‌تر'],
            'lift': ['لیفت', 'بالا', 'بالا بردن', 'کشیدن بالا']
        }
        
        self.STYLE_KEYWORDS = {
            'lip': {'قلوه‌ای': 'heart_shape', 'روسی': 'russian', 'طبیعی': 'natural'},
            'nose': {'قلمی': 'slim_bridge', 'عروسکی': 'doll_tip', 'طبیعی': 'natural'},
        }
        
        logger.info("✅ BeautyEngineModel ready")

    def process(self, image: np.ndarray, text: str, intensity: Optional[float] = None,
                show_area: bool = False, blend_method: str = 'alpha') -> Dict:
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
        result = self.blending.blend(image, warped, mask, method=blend_method)

        if show_area:
            result = self.warping.draw_area(result, points, parsed['area'])

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
        text_lower = text.lower()
        result = {'area': None, 'action': None, 'intensity': 0.5}

        for area, keywords in self.AREA_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                result['area'] = area
                break

        if result['area'] and result['area'] in self.STYLE_KEYWORDS:
            styles = self.STYLE_KEYWORDS[result['area']]
            for kw, action in styles.items():
                if kw in text_lower:
                    result['action'] = action
                    break

        if not result['action']:
            for action, keywords in self.ACTION_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    result['action'] = action
                    break

        if 'کم' in text_lower or 'ملایم' in text_lower:
            result['intensity'] = 0.3
        elif 'زیاد' in text_lower or 'خیلی' in text_lower:
            result['intensity'] = 0.8

        match = re.search(r'(\d+)\s*(cc|سیسی|ml)', text.translate(self.PERSIAN_DIGITS), re.I)
        if match:
            result['intensity'] = min(float(match.group(1)) * 0.6, 1.0)

        return result

    def _generate_description(self, parsed: Dict, intensity: float) -> str:
        area_names = {'nose': 'بینی', 'lip': 'لب', 'jaw': 'فک', 'cheek': 'گونه', 'eye': 'چشم'}
        action_names = {
            'smaller': 'کوچک‌تر', 'bigger': 'بزرگ‌تر', 'fuller': 'پرتر',
            'sharper': 'تیزتر', 'lift': 'لیفت', 'heart_shape': 'قلوه‌ای',
            'slim_bridge': 'قلمی', 'doll_tip': 'عروسکی'
        }
        area = area_names.get(parsed.get('area', ''), 'ناحیه')
        action = action_names.get(parsed.get('action', ''), 'تغییر')
        return f"{area} با شدت {int(intensity*100)}% {action} می‌شود"


beauty_engine = BeautyEngineModel()