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

        # ============================================
        # دیکشنری‌های داخلی — کامل با انواع بینی و لب
        # ============================================
        self.AREA_KEYWORDS = {
            'nose': ['بینی', 'دماغ', 'دماغم', 'بینی من', 'دماغ من', 'نوک بینی', 'پل بینی', 'منخر'],
            'lip': ['لب', 'لبام', 'لبامو', 'لب من', 'دهان', 'لب بالا', 'لب پایین', 'گوشه لب', 'کمان کوپید'],
            'jaw': ['فک', 'چانه', 'فکم', 'چانم', 'چونه', 'چونم', 'چونه‌ام', 'ذقن', 'فک من', 'خط فک'],
            'cheek': ['گونه', 'گونه‌ها', 'گونه من', 'استخوان گونه'],
            'forehead': ['پیشانی', 'پیشونم', 'پیشانی من'],
            'eye': ['چشم', 'چشمام', 'چشم من', 'چشم‌ها', 'پلک']
        }

        self.ACTION_KEYWORDS = {
            'smaller': ['کوچیک‌تر', 'کوچک‌تر', 'کوچیک', 'کوچک', 'باریک‌تر', 'کم‌تر', 'کم', 'نازک‌تر', 'کاهش', 'ریزتر'],
            'bigger': ['بزرگ‌تر', 'بلندتر', 'بیشتر', 'بزرگ', 'درشت‌تر', 'پهن‌تر', 'افزایش'],
            'fuller': ['پرتر', 'حجم‌تر', 'پرشد', 'برجسته‌تر', 'حجیم‌تر', 'پر', 'ژل', 'ژل زده', 'تزریق'],
            'sharper': ['تیزتر', 'مشخص‌تر', 'زاویه‌دارتر'],
            'smoother': ['صاف‌تر', 'یکدست‌تر', 'نرم‌تر'],
            'rounder': ['گردتر', 'گردتر کن', 'گِرد'],
            'lift': ['بالا ببر', 'بالا کن', 'ببر بالا', 'کن بالا', 'لیفت', 'کشیدن بالا'],
            # 🆕 ابعاد دقیق
            'narrower': ['باریک', 'باریک‌تر'],
            'wider': ['پهن'],
            'shorter': ['کوتاه', 'کوتاه‌تر', 'کوتاه کن'],
            'longer': ['بلند', 'طولانی'],
        }

        # ============================================
        # 🆕 انواع بینی — دیکشنری کامل کلینیکی + عامیانه
        #    (استایل‌های زیبایی + فیلر + جزئیات غضروف/پوست)
        # ============================================
        self.NOSE_STYLES = {
            # ---- فرم‌های کلی (درخواست رایج) ----
            'fleshy': ['گوشتی', 'مگسکی', 'پر پهن', 'پرپهن', 'گوشتالو'],
            'fantasy': ['فانتزی', 'اروپایی'],
            'natural': ['طبیعی', 'عادی'],
            # ⭐ بهترین حالت فوق‌واقعی — موتور هوشمند شخصی‌سازی‌شده
            'ideal_realistic': ['بهترین حالت', 'بهترین بینی', 'بهترینش',
                                'خیلی واقعی', 'فوق واقعی', 'فوق‌واقعی',
                                'واقعی ترین', 'واقعی‌ترین', 'ایده آل',
                                'ایده‌آل', 'ایده آل ترین', 'مدل عمل شده',
                                'بینی عمل شده', 'عمل شده خوب', 'سوپر رئال',
                                'سوپررئال', 'طبیعی ترین', 'طبیعی‌ترین'],
            'bony': ['استخوانی', 'تیزی', 'استخونی'],
            'half_fantasy': ['نیمه فانتزی', 'نیمه‌فانتزی', 'نصف فانتزی'],
            'doll_tip': ['عروسکی', 'مزدانه', 'مزدانة', 'دخترونه'],
            'upturned_tip': ['نوک بالا', 'نوکش بالا', 'نوکشو بالا', 'سر بالا',
                             'سربالا', 'بالای نوک', 'نوک بالا رفته', 'نوک بالا ببر',
                             'نوکشو بیار بالا', 'نوک بینی بالا'],
            'droopy_tip': ['نوک پایین', 'نوکش پایین', 'نوک افتاده'],
            'filler': ['فیلر زده', 'فیلر خورده', 'فیلر بینی', 'ژل بینی', 'فیلر',
                       'تزریق بینی', 'پر شدن گودی بینی'],
            'slim_bridge': ['قلمی', 'قلم مانند', 'باریک قلمی'],

            # ---- جزئیات جراحی/غضروف (دقیق) ----
            'hump_reduction': ['قوز بینی', 'قوزش', 'قوز رو', 'قوزشو', 'بدون قوز',
                               'قوز بهتر', 'قوز', 'قوس بینی', 'برداشتن قوز',
                               'قوزشو بردار', 'قوزشو بگیر'],
            'dorsum_smooth': ['قوس صاف', 'صاف کردن قوس', 'خط بینی صاف', 'صافشه'],
            'tip_refinement': ['نوک ظریف', 'نوک رو جمع کن', 'نوک تیز بشه',
                               'غضروف نوک', 'تعریف نوک', 'نوک مشخص بشه'],
            'alar_reduction': ['بال بینی کوچیک', 'بالی', 'بال‌هاشو جمع کن',
                               'سوراخ بینی کوچیک', 'سوراخ‌ها کوچیک',
                               'پهنای سوراخ', 'فلر بینی', 'بال بینی'],
            'nostril_show': ['سوراخ معلوم', 'سوراخ دیده نشه', 'سوراخ کم'],
            'columella_show': ['ستون بینی', 'کولوملا'],
            'deviation_fix': ['کجی بینی', 'بینیم کجه', 'کج شده', 'انحراف بینی',
                              'متقارن کن', 'تقارن بینی'],
            'tip_projection': ['نوک جلو', 'نوک عقب', 'برجسته نوک', 'پروجکشن'],
            'nasal_length': ['طول بینی', 'بلندی بینی'],
            'nasal_base': ['پایه بینی', 'قاعده بینی'],
            'supratip_break': ['گودی بالای نوک', 'سوپراتایپ'],
        }

        # ============================================
        # 🆕 انواع لب — ۱۰ استایل
        # ============================================
        self.LIP_STYLES = {
            'russian': ['روسی', 'روس'],
            'brazilian': ['برزیلی', 'برزیل'],
            'hollywood': ['هالیوودی', 'هالیود', 'هالیوود'],
            'heart_shape': ['قلوه‌ای', 'قلب'],
            'classic': ['کلاسیک'],
            'natural': ['طبیعی'],
            'cupids_bow': ['کمان کوپید', 'کوپید'],
            'corner_lift': ['گوشه لب بالا', 'لبخند لب', 'گوشه‌ها بالا'],
        }

        logger.info("✅ BeautyEngineModel ready")

    def process(self, image: np.ndarray, text: str, intensity: Optional[float] = None,
                show_area: bool = False, blend_method: str = 'alpha') -> Dict:
        landmarks = self.face_parser.detect_from_image(image)
        if not landmarks:
            return {'status': 'error', 'message': 'چهره‌ای شناسایی نشد'}

        # 🆕 تشخیص نمای چهره (روبرو / نیم‌رخ چپ / نیم‌رخ راست)
        view = self._detect_view(landmarks, image.shape)

        # 🆕 پارسر چند-تغییره: «باریک شود و قوز بهتر شود» → [narrower, hump_reduction]
        changes = self._parse_multi(text)
        if not changes:
            return {'status': 'error', 'message': f'متوجه نشدم: "{text}"'}

        result = image.copy()
        applied = []
        for parsed in changes:
            area, action = parsed['area'], parsed['action']
            points = self.face_parser.get_points(landmarks, area)
            if not points:
                continue

            inten = intensity or parsed.get('intensity', 0.5)
            # 🆕 نیم‌رخ: چرخش‌های نوک حساس‌ترند — کمی مهار تا سایه/کشیدگی نیفتد
            if area == 'nose' and view != 'front':
                inten = round(min(inten * 0.8, 1.0), 3)

            warped = self.warping.warp(result, points, inten, action, area,
                                       landmarks=landmarks,
                                       image_shape=image.shape)

            # 🆕 ضدسایه: تطبیق روشنایی ناحیه قبل از ترکیب
            if area == 'nose':
                warped = self._fix_shadow_artifacts(result, warped, points)

            mask = self.warping.create_mask(result, points)

            # 🆕 برای بینی: پایان‌بندی Poisson (حذف هرگونه درز روشنایی)
            if area == 'nose':
                merged = self._poisson_finish(result, warped, mask)
                result = merged if merged is not None else \
                    self.blending.blend(result, warped, mask, method=blend_method)
            else:
                result = self.blending.blend(result, warped, mask,
                                             method=blend_method)
            applied.append(parsed)

        if not applied:
            return {'status': 'error', 'message': f'متوجه نشدم: "{text}"'}

        # 🆕 گزارش موتور هوشمند «بهترین حالت» — برای نمایش تحلیل در UI
        ai_report = None
        if any(a.get('action') == 'ideal_realistic' for a in applied):
            try:
                from warping.ideal_nose_ai import IDEAL_NOSE_AI
                ai_report = getattr(IDEAL_NOSE_AI, 'last_report', None)
            except Exception:
                ai_report = None

        if show_area and applied:
            points = self.face_parser.get_points(landmarks, applied[0]['area'])
            result = self.warping.draw_area(result, points, applied[0]['area'])

        return {
            'status': 'success',
            'image': result,
            'original': image,
            'changes': applied[0] if len(applied) == 1 else {
                'area': applied[0]['area'],
                'actions': [a['action'] for a in applied],
                'intensity': intensity or applied[0].get('intensity', 0.5),
            },
            'applied_changes': applied,
            'view': view,
            'intensity': intensity or applied[0].get('intensity', 0.5),
            # 🆕 تحلیل هوشمند «بهترین حالت» (قبل/بعد/برنامه اجراشده)
            'ai_report': ai_report,
            'description': ' + '.join(self._generate_description(a,
                intensity or a.get('intensity', 0.5)) for a in applied),
            'message': '✅ تغییرات اعمال شد'
        }

    # ============================================
    # 🆕 تشخیص نمای چهره — yaw از هندسه چشم/صورت
    # ============================================
    def _detect_view(self, landmarks, shape) -> str:
        """front | left_profile | right_profile — از نسبت گوشه چشم به لبه صورت."""
        try:
            if len(landmarks) < 468:
                return 'front'
            h, w = shape[:2]
            L = landmarks[33]   # گوشه داخلی چشم چپ
            R = landmarks[263]  # گوشه داخلی چشم راست
            earL = landmarks[127]  # لبه چپ صورت
            earR = landmarks[356]  # لبه راست صورت
            dL = abs(L['x'] - earL['x'])
            dR = abs(earR['x'] - R['x'])
            total = dL + dR
            if total < 1e-3:
                return 'front'
            ratio = dL / total
            if ratio > 0.62:
                return 'right_profile'   # چشم چپ نزدیک لبه → سر به راست چرخیده
            if ratio < 0.38:
                return 'left_profile'
            return 'front'
        except Exception:
            return 'front'

    # ============================================
    # 🆕 ضدسایه — رفع سایه کاذب بعد از چرخش نوک
    # ============================================
    def _fix_shadow_artifacts(self, original, warped, points):
        """
        وقتی نوک بالا می‌رود، بافت جابه‌جا می‌شود و روشنایی محلی به‌هم
        می‌ریزد → سایه کاذب. راه‌حل: تطبیق میانگین/انحراف روشنایی (L)
        ناحیه warp شده با تصویر اصلی، فقط داخل ماسک.
        """
        try:
            import cv2 as _cv2
            mask = np.zeros(original.shape[:2], np.uint8)
            _cv2.fillPoly(mask, [np.array(points, np.int32)], 255)

            orig_lab = _cv2.cvtColor(original, _cv2.COLOR_BGR2LAB)
            warp_lab = _cv2.cvtColor(warped, _cv2.COLOR_BGR2LAB)

            m = (mask > 0)
            if m.sum() < 50:
                return warped

            # آماره روشنایی قبل/بعد داخل ناحیه
            L_o = orig_lab[..., 0].astype(np.float32)
            L_w = warp_lab[..., 0].astype(np.float32)
            mean_o, std_o = float(L_o[m].mean()), float(L_o[m].std() + 1e-6)
            mean_w, std_w = float(L_w[m].mean()), float(L_w[m].std() + 1e-6)

            # نگاشت خطی ملایم (حداکثر ۱۲٪ اصلاح تا طبیعی بماند)
            gain = float(np.clip(std_o / std_w, 0.88, 1.12))
            bias = float(np.clip(mean_o - mean_w * gain, -14, 14))

            L_new = np.clip(L_w + (mean_o - mean_w), 0, 255).astype(np.uint8)

            warp_lab[..., 0] = L_new
            fixed = _cv2.cvtColor(warp_lab, _cv2.COLOR_LAB2BGR)

            # ترکیب: فقط داخل ماسک اصلاح شود
            m3 = (mask.astype(np.float32) / 255.0)[..., None]
            out = (warped.astype(np.float32) * (1 - m3) +
                   fixed.astype(np.float32) * m3)
            return np.clip(out, 0, 255).astype(np.uint8)
        except Exception:
            return warped

    def _poisson_finish(self, original, warped, mask):
        """SeamlessClone فقط برای درز مرزی — با ماسک تنگ تا اثر warp حفظ شود.
        نکته: seamlessClone با ماسک پهن، تغییرات هندسی را بازتوزیع می‌کند
        (۹۰٪ افت اثر!). پس فقط هسته تنگ ماسک را کلون می‌کنیم و لبه را از
        خودِ warped می‌گیریم."""
        try:
            import cv2 as _cv2
            m = (mask > 10).astype(np.uint8)
            ys, xs = np.where(m > 0)
            if len(ys) < 100:
                return None
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x1 - x0 < 8 or y1 - y0 < 8:
                return None
            center = ((x0 + x1) // 2, (y0 + y1) // 2)

            # 🎯 ماسک تنگ: erode تا فقط داخل ناحیه کلون شود؛
            # سپس خروجی کلون را با ماسک نرم اصلی روی warped برگردانیم
            tight = _cv2.erode(mask, np.ones((7, 7), np.uint8), iterations=1)
            mixed = _cv2.seamlessClone(
                warped, original, tight, center, _cv2.NORMAL_CLONE)

            # ترکیب نهایی: هسته از کلون (درز صفر)، حاشیه از warped (فرم حفظ)
            m_soft = mask.astype(np.float32) / 255.0
            m3 = m_soft[..., None]
            out = (mixed.astype(np.float32) * m3 +
                   warped.astype(np.float32) * (1 - m3))
            return np.clip(out, 0, 255).astype(np.uint8)
        except Exception:
            return None

    # ============================================
    # 🆕 پارسر چند-تغییره
    # «بینی رو باریک کن و قوزشم بردار» → [narrower, hump_reduction]
    # ============================================
    def _parse_multi(self, text: str) -> list:
        import re as _re
        t = text.strip()
        if not t:
            return []
        # شکستن روی: «و»، ویرگول/نقطه-ویرگول، «بعد»، «همچنین»، «+»
        parts = _re.split(r'\s+و\s+|[,،؛;]\s*|\s+همچنین\s+|\s+بعد\s+|\+', t)
        parts = [p for p in (x.strip() for x in parts) if p]

        results = []
        seen = set()
        for part in parts:
            r = self._parse_request(part)
            if r.get('area') and r.get('action'):
                key = (r['area'], r['action'])
                if key not in seen:
                    seen.add(key)
                    results.append(r)

        # fallback: اگر شکستن چیزی پیدا نکرد، کل متن را یکجا پارس کن
        if not results:
            r = self._parse_request(t)
            if r.get('area') and r.get('action'):
                results.append(r)
        return results

    def _parse_request(self, text: str) -> Dict:
        text_lower = text.lower()
        result = {'area': None, 'action': None, 'style': None, 'intensity': 0.5}

        # ---------- ۱. ناحیه ----------
        for area, keywords in self.AREA_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                result['area'] = area
                break

        # اگر ناحیه صریح نبود ولی کلمهٔ استایل بینی/لب بود، ناحیه را حدس بزن
        if not result['area']:
            if any(kw for kws in self.NOSE_STYLES.values() for kw in kws if kw in text_lower):
                result['area'] = 'nose'
            elif any(kw for kws in self.LIP_STYLES.values() for kw in kws if kw in text_lower):
                result['area'] = 'lip'

        # ---------- ۲. استایل اختصاصی ناحیه ----------
        style_map = {}
        if result['area'] == 'nose':
            style_map = self.NOSE_STYLES
        elif result['area'] == 'lip':
            # استایل‌های لب + لیفت گوشه لب (که در NOSE_STYLES هم هست)
            style_map = self.LIP_STYLES
            style_map = {**style_map,
                         'corner_lift': self.NOSE_STYLES.get('corner_lift', [])}

        if style_map:
            best_kw, best_len, best_action = None, 0, None
            for action, keywords in style_map.items():
                for kw in keywords:
                    if kw in text_lower and len(kw) > best_len:
                        best_kw, best_len, best_action = kw, len(kw), action
            if best_action:
                result['action'] = best_action
                result['style'] = best_action

        # ---------- ۳. action کلی (اگر استایل نبود) ----------
        if not result['action']:
            for action, keywords in self.ACTION_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    result['action'] = action
                    break

        # 🆕 نگاشت action های عمومی به استایل‌های بینی
        if result['area'] == 'nose' and 'بالا' in text_lower:
            # بینی + «بالا بردن» → نوک بالا (استایل upturned_tip)
            result['action'] = 'upturned_tip'
        elif result['area'] == 'lip' and result['action'] == 'lift':
            result['action'] = 'corner_lift'
        elif result['area'] == 'nose' and result['action'] in ('smaller', 'bigger'):
            if result['action'] == 'smaller' and ('باریک' in text_lower or 'نازک' in text_lower):
                result['action'] = 'narrower'
            elif result['action'] == 'smaller' and ('کوتاه' in text_lower):
                result['action'] = 'shorter'
            elif result['action'] == 'bigger' and 'پهن' in text_lower:
                result['action'] = 'wider'

        # 🆕 فک: «تیز کن» → sharper (نه smaller)
        if result['area'] == 'jaw' and result['action'] in ('smaller', None):
            if any(kw in text_lower for kw in ['تیز', 'زاویه', 'مشخص', 'خط فک']):
                result['action'] = 'sharper'

        # 🆕 لیفت عمومی روی لب → گوشه‌ها بالا
        if result['area'] == 'lip' and result['action'] == 'lift':
            result['action'] = 'corner_lift'

        # ---------- ۴. شدت ----------
        if 'کم' in text_lower or 'ملایم' in text_lower:
            result['intensity'] = 0.3
        elif 'زیاد' in text_lower or 'خیلی' in text_lower:
            result['intensity'] = 0.8

        # ---------- ۵. مقدار عددی (cc/ml) ----------
        match = re.search(r'(\d+)\s*(cc|سیسی|ml)', text.translate(self.PERSIAN_DIGITS), re.I)
        if match:
            result['intensity'] = min(float(match.group(1)) * 0.6, 1.0)

        return result

    def _generate_description(self, parsed: Dict, intensity: float) -> str:
        area_names = {'nose': 'بینی', 'lip': 'لب', 'jaw': 'فک', 'cheek': 'گونه',
                      'forehead': 'پیشانی', 'eye': 'چشم'}
        action_names = {
            'smaller': 'کوچک‌تر', 'bigger': 'بزرگ‌تر', 'fuller': 'پرتر',
            'sharper': 'تیزتر', 'lift': 'لیفت', 'smoother': 'صاف‌تر',
            'heart_shape': 'قلوه‌ای', 'slim_bridge': 'قلمی', 'doll_tip': 'عروسکی',
            'russian': 'روسی', 'brazilian': 'برزیلی', 'hollywood': 'هالیوودی',
            'classic': 'کلاسیک', 'cupids_bow': 'کمان کوپید تیز',
            'corner_lift': 'لیفت گوشه لب',
            # بینی
            'fleshy': 'گوشتی', 'fantasy': 'فانتزی', 'half_fantasy': 'نیمه‌فانتزی',
            'bony': 'استخوانی', 'natural': 'طبیعی', 'upturned_tip': 'نوک بالا',
            'filler': 'فیلر زده شده',
            'ideal_realistic': '⭐ بهترین حالت فوق‌واقعی (تحلیل هوشمند چهره)',
            'narrower': 'باریک‌تر', 'wider': 'پهن‌تر',
            'shorter': 'کوتاه‌تر', 'longer': 'بلندتر',
            'droopy_tip': 'نوک افتاده', 'hump_reduction': 'برداشتن قوز',
        }
        area = area_names.get(parsed.get('area', ''), 'ناحیه')
        action = action_names.get(parsed.get('action', ''), 'تغییر')
        return f"{area} با شدت {int(intensity*100)}% {action} می‌شود"


beauty_engine = BeautyEngineModel()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing BeautyEngine")
    print("=" * 60)

    test_texts = [
        "دماغم رو کوچیکتر کن",
        "بینی گوشتی",
        "بینی فانتزی",
        "بینی طبیعی",
        "بینی استخوانی",
        "نیمه فانتزی",
        "بینی عروسکی مزدانه",
        "نوک بینی رو بالا ببر",
        "بینی فیلر زده شده",
        "بینی قلمی",
        "لب روسی",
        "لب برزیلی",
        "لب هالیوودی",
        "لبام قلوه‌ای با ۲ سی‌سی",
        "فکم رو تیز کن",
        "لب هام رو پرتر کن",
    ]

    for text in test_texts:
        r = beauty_engine._parse_request(text)
        style = f" [استایل]" if r.get('style') else ""
        print(f"📝 '{text}' → area={r.get('area')}, action={r.get('action')}{style}, شدت={r.get('intensity')}")
