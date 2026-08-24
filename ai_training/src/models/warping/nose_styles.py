"""
موتور Warping آناتومیک بینی — سطح جراحی رینوپلاستی
====================================================
هر استایل = دنباله‌ای از «مانورهای آناتومیک» روی اجزای واقعی بینی:
  • tip_rotation      چرخش نوک حول محور پل (زاویه‌ای، نه جابجایی آزاد)
  • tip_projection    برجستگی نوک در راستای محور رادیکس→نوک
  • dorsum_reshape    بازسازی قوس پل (قوز/گودی/صاف)
  • alar_narrowing    باریک‌سازی بال‌ها (Weir-style، پایه ثابت)
  • columella_set     تنظیم کولوملا (زاویه nasolabial)
  • radix_fill        پرکردن رادیکس (فیلر)

همه شدت‌ها با beauty_standards سقف‌خورده‌اند؛ هیچ مانوری اجازه ندارد
پیکسل بیرون ناحیه را بکشد (map clip + ماسک دو مرحله‌ای).
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    from ...beauty_standards import clamp_intensity, NOSE_CANON
    from .nose_anatomy import NoseAnatomy
except Exception:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.dirname(__file__))
    from beauty_standards import clamp_intensity, NOSE_CANON
    from nose_anatomy import NoseAnatomy

import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# دیتکتور تنبل برای حلقه راستی‌آزمایی موتور AI
# (اولین بار ساخته می‌شود؛ اگر مدپایپ نبود None برمی‌گردد)
# ---------------------------------------------------------
_AI_DETECTOR = None


def _ai_detector():
    global _AI_DETECTOR
    if _AI_DETECTOR is None:
        try:
            try:
                from ..face_parser.model import FaceParserModel
            except ImportError:
                from face_parser.model import FaceParserModel
            _AI_DETECTOR = FaceParserModel().detect_from_image
        except Exception as e:
            logger.warning(f"AI verify detector unavailable: {e}")
            _AI_DETECTOR = False
    return _AI_DETECTOR or None


# ============================================
#   پایه: warp ایمن
# ============================================

def _roi_bounds(shape, pts: np.ndarray, pad_frac: float = 0.40):
    h, w = shape[:2]
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0)
    pad = int(r * pad_frac)
    rx0 = max(0, int(cx - r / 2 - pad)); ry0 = max(0, int(cy - r / 2 - pad))
    rx1 = min(w, int(cx + r / 2 + pad)); ry1 = min(h, int(cy + r / 2 + pad))
    return (rx0, ry0, rx1, ry1) if rx1 - rx0 >= 4 and ry1 - ry0 >= 4 else (0, 0, w, h)


def _cos_win(dist, radius):
    u = np.clip(dist / max(radius, 1e-6), 0, 1)
    return (0.5 * (1 + np.cos(np.pi * u))).astype(np.float32)


def _warp_field(image, pts, center, radius, dx=0.0, dy=0.0,
                sx=1.0, sy=1.0, pivot=None, weight=None):
    """یک پاس warp نرم: جابجایی + مقیاس با پنجره کسینوسی. map همیشه داخل ROI."""
    if radius <= 1:
        return image
    p = center if pivot is None else pivot
    x0, y0, x1, y1 = _roi_bounds(image.shape, pts)

    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    vx = xs - p[0]; vy = ys - p[1]
    dist = np.sqrt(vx * vx + vy * vy)
    win = _cos_win(dist, radius)
    if weight is not None:
        win = win * weight

    fx = 1 + (sx - 1) * win
    fy = 1 + (sy - 1) * win
    map_x = p[0] + vx * fx - dx * win - x0
    map_y = p[1] + vy * fy - dy * win - y0

    map_x = np.clip(map_x, 0, x1 - x0 - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, y1 - y0 - 1).astype(np.float32)

    warped = cv2.remap(image[y0:y1, x0:x1], map_x, map_y,
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    out = image.copy()
    m3 = win[..., None]
    out[y0:y1, x0:x1] = (warped * m3 + out[y0:y1, x0:x1] * (1 - m3)).astype(np.uint8)
    return out


def _sharpen_region(image, mask, amount=0.3):
    blur = cv2.GaussianBlur(image, (0, 0), 2.0)
    sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
    m3 = mask[..., None]
    return np.clip(image * (1 - m3) + sharp * m3, 0, 255).astype(np.uint8)


# ============================================
#   مانورهای آناتومیک
# ============================================

class Maneuvers:

    @staticmethod
    def tip_rotation(image, anat: NoseAnatomy, degrees: float) -> np.ndarray:
        """چرخش نوک حول محور پایه→نوک (نسخه ضد-اعوجاج).

        اصول:
          • pivot = میانه دو بال (پایه واقعی بینی)
          • پنجره بیضی‌وار هم‌راستا با محور بینی (نه دایره بزرگ)
          • زاویه مؤثر هر پیکسل هرگز از سقف زاویه عبور نمی‌کند
          • ROI بزرگ‌تر تا جابجایی نوک داخل کراپ بماند (بدون پارگی clip)
        """
        pivot_raw = anat.get('radix')
        tip = anat.get('tip')
        if pivot_raw is None or tip is None:
            return image
        alar_mid_x = float(anat.alar_mid[0])
        pivot = np.array([alar_mid_x, pivot_raw[1]], dtype=np.float32)
        axis = tip - pivot
        height = float(np.linalg.norm(axis)) + 1e-6
        u = axis / height
        # بردار عمود بر محور
        n = np.array([-u[1], u[0]], dtype=np.float32)

        pts = anat.ordered_array()
        if pts is None:
            return image
        x0, y0, x1, y1 = _roi_bounds(image.shape, pts, pad_frac=0.75)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        dx = xs - pivot[0]; dy = ys - pivot[1]
        along = (dx * u[0] + dy * u[1]) / height            # 0=پایه، 1=نوک
        dist = np.sqrt(dx * dx + dy * dy)

        # وزن محوری: فقط نیمه بالایی بینی می‌چرخد؛ نوک کامل
        axial_raw = np.clip(along / 0.45, 0, 1)
        axial = axial_raw * axial_raw * (3 - 2 * axial_raw)  # smoothstep
        # پشت نوک (along>1.15) وزن جمع شود تا پیشانی کشیده نشود
        over = np.clip((along - 1.15) / 0.35, 0, 1)
        axial = axial * (1 - over * 0.9)

        # پنجره شعاعی محدودتر (قبلاً 2.2h بود → کشیدگی گونه)
        win = _cos_win(dist, height * 1.35)
        # پهنه جانبی باریک‌تر و هم‌مرکز با محور
        perp = dx * n[0] + dy * n[1]
        half_w = max(anat.nasal_width * 0.85, height * 0.42)
        lateral = _cos_win(np.abs(perp), half_w)

        w = axial * win * lateral
        peak = float(w.max()) if w.size else 0.0
        if peak > 1e-6:
            w = w * (0.35 / peak) if peak < 0.35 else w  # کف مؤثر برای حس چرخش

        ang = np.radians(degrees) * w
        ca, sa = np.cos(ang), np.sin(ang)
        rx = dx * ca - dy * sa
        ry = dx * sa + dy * ca

        map_x = pivot[0] + rx - x0
        map_y = pivot[1] + ry - y0
        # حاشیه امن: فقط در لبه بیرونی crop محدود شود (نه وسط ناحیه)
        map_x = np.clip(map_x, -1, x1 - x0).astype(np.float32)
        map_y = np.clip(map_y, -1, y1 - y0).astype(np.float32)
        np.clip(map_x, 0, x1 - x0 - 1, out=map_x)
        np.clip(map_y, 0, y1 - y0 - 1, out=map_y)

        warped = cv2.remap(image[y0:y1, x0:x1], map_x, map_y,
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        out = image.copy()
        m3 = w[..., None]
        out[y0:y1, x0:x1] = (warped * m3 + out[y0:y1, x0:x1] * (1 - m3)).astype(np.uint8)
        return out

    @staticmethod
    def tip_projection(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """برجستگی/عقب‌رفتن نوک در راستای محور بینی (2D: در راستای رادیکس→نوک)."""
        tip = anat.get('tip')
        if tip is None or anat.get('radix') is None:
            return image
        u = anat.tip_projection_axis
        pts = anat.ordered_array()
        if pts is None:
            return image
        return _warp_field(image, pts, tip, anat.nasal_height * 0.45,
                           dx=u[0] * amount, dy=u[1] * amount,
                           pivot=anat.get('radix'))

    @staticmethod
    def dorsum_reshape(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """قوس پل: amount>0 پهن/قوز، <0 باریک/صاف. نوار مرکزی بین radix تا tip."""
        mid = anat.get('mid_bridge')
        pts = anat.ordered_array()
        if mid is None or pts is None:
            return image
        h = anat.nasal_height
        # ✅ pivot = radix؛ شعاع کل طول پل تا کمی جلوتر از نوک
        return _warp_field(image, pts, mid, h * 0.95,
                           sx=1 + amount, sy=1.0, pivot=anat.get('radix'))

    @staticmethod
    def radix_fill(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """پر کردن گودی رادیکس (فیلر پل)."""
        radix = anat.get('radix')
        pts = anat.ordered_array()
        if radix is None or pts is None:
            return image
        u = anat.tip_projection_axis
        return _warp_field(image, pts, radix, anat.nasal_height * 0.35,
                           dx=-u[0] * amount * 0.6, dy=-u[1] * amount * 0.6,
                           sx=1 + amount * 0.5, pivot=radix)

    @staticmethod
    def alar_narrowing(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """باریک‌سازی بال‌ها: هر بال به سمت خط وسط؛ پایه‌ها ثابت (Weir).
        🎯 شعاع محدود به ناحیه بال (نه 0.75*w) تا نوک/پل کشیده نشود —
        قبلاً این باعث کج شدن نوک بعد از چرخش بود."""
        pts = anat.ordered_array()
        if pts is None or anat.get('alar_l') is None or anat.get('alar_r') is None:
            return image
        result = image
        w = anat.nasal_width
        for side, anchor in (('alar_l', anat.get('alar_l')),
                             ('alar_r', anat.get('alar_r'))):
            direction = 1.0 if side == 'alar_l' else -1.0   # به داخل
            result = _warp_field(result, pts, anchor, w * 0.38,
                                 dx=direction * amount, sy=1.0)
        return result

    @staticmethod
    def alar_flaring(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        pts = anat.ordered_array()
        if pts is None or anat.get('alar_l') is None or anat.get('alar_r') is None:
            return image
        result = image
        w = anat.nasal_width
        for side, anchor in (('alar_l', anat.get('alar_l')),
                             ('alar_r', anat.get('alar_r'))):
            direction = -1.0 if side == 'alar_l' else 1.0   # به بیرون
            result = _warp_field(result, pts, anchor, w * 0.38,
                                 dx=direction * amount, sy=1.0)
        return result

    @staticmethod
    def columella_set(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """تنظیم کولوملا: amount>0 پایین (زاویه nasolabial بسته‌تر)، <0 بالا."""
        col = anat.get('columella')
        pts = anat.ordered_array()
        if col is None or pts is None or anat.get('tip') is None:
            return image
        return _warp_field(image, pts, col, anat.nasal_height * 0.30,
                           dy=amount, pivot=anat.get('tip'))

    @staticmethod
    def nostril_symmetry(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """تقارن سوراخ‌ها: هر دو به مرکز پایه نزدیک/دور می‌شوند (فقط تصحیح ملایم)."""
        base = anat.get('base_center')
        pts = anat.ordered_array()
        if base is None or pts is None:
            return image
        result = image
        for name in ('nostril_l', 'nostril_r'):
            anchor = anat.get(name)
            if anchor is None:
                continue
            to_center = base - anchor
            result = _warp_field(result, pts, anchor, anat.nasal_width * 0.6,
                                 dx=to_center[0] * amount * 0.3,
                                 dy=to_center[1] * amount * 0.3)
        return result

    @staticmethod
    def sidewall_definition(image, anat: NoseAnatomy, amount: float) -> np.ndarray:
        """تعریف دیواره‌ها: باریک‌سازی ملایم دو نوار کناری."""
        pts = anat.ordered_array()
        if pts is None:
            return image
        result = image
        w = anat.nasal_width
        for name in ('sidewall_l', 'sidewall_r'):
            anchor = anat.get(name)
            if anchor is None:
                continue
            inward = 1.0 if name == 'sidewall_l' else -1.0
            result = _warp_field(result, pts, anchor, w * 0.5,
                                 dx=inward * amount * 0.5)
        return result


# ============================================
#   استایل‌ها — ترکیب مانورها طبق اصول جراحی
# ============================================

def _resolve(landmarks, shape) -> Optional[NoseAnatomy]:
    """ساخت NoseAnatomy از لندمارک کامل (پیکسلی یا نرمال) یا پلی‌گان ۱۸ نقطه‌ای."""
    if (isinstance(landmarks, list) and landmarks
            and len(landmarks) >= 468 and shape is not None):
        h, w = shape[:2]
        first = landmarks[0]
        # تشخیص مختصات نرمال (0..1) در برابر پیکسلی
        if 0.0 <= float(first['x']) <= 1.0 and 0.0 <= float(first['y']) <= 1.0:
            # نرمال → به پیکسل تبدیل کن (کپی تا ورودی اصلی تغییر نکند)
            px = [{'x': lm['x'] * w, 'y': lm['y'] * h, 'z': lm.get('z', 0.0)}
                  for lm in landmarks]
            a = NoseAnatomy(landmarks=px, image_shape=shape)
        else:
            a = NoseAnatomy(landmarks=landmarks, image_shape=shape)
        if a.valid:
            return a
    poly = landmarks
    if poly is not None and not isinstance(poly, list) and hasattr(poly, 'tolist'):
        poly = poly.tolist()
    if isinstance(poly, list) and len(poly) == 18:
        a = NoseAnatomy(fallback_polygon=poly)
        if a.valid:
            return a
    return None


class NoseAnatomyStyles:
    """امضا: (image, landmarks, image_shape, intensity_raw) → image"""

    # ---------- اندازه ----------
    @staticmethod
    def smaller(image, landmarks, shape, intensity):
        i = clamp_intensity('smaller', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = image
        result = Maneuvers.alar_narrowing(result, a, a.nasal_width * 0.34 * i)
        result = Maneuvers.dorsum_reshape(result, a, -0.10 * i)
        result = Maneuvers.tip_projection(result, a, -a.nasal_height * 0.09 * i)
        result = Maneuvers.columella_set(result, a, -a.nasal_height * 0.05 * i)
        return result

    @staticmethod
    def bigger(image, landmarks, shape, intensity):
        i = clamp_intensity('bigger', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.alar_flaring(image, a, a.nasal_width * 0.16 * i)
        result = Maneuvers.dorsum_reshape(result, a, 0.08 * i)
        return result

    @staticmethod
    def narrower(image, landmarks, shape, intensity):
        i = clamp_intensity('narrower', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        return Maneuvers.alar_narrowing(image, a, a.nasal_width * 0.42 * i)

    @staticmethod
    def wider(image, landmarks, shape, intensity):
        i = clamp_intensity('wider', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        return Maneuvers.alar_flaring(image, a, a.nasal_width * 0.36 * i)

    @staticmethod
    def shorter(image, landmarks, shape, intensity):
        """کوتاه: چرخش نوک بالا (کاهش ارتفاع واقعی) + جمع کولوملا."""
        i = clamp_intensity('shorter', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.tip_rotation(image, a, 10.0 * i)
        result = Maneuvers.columella_set(result, a, -a.nasal_height * 0.05 * i)
        return result

    @staticmethod
    def longer(image, landmarks, shape, intensity):
        i = clamp_intensity('longer', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.tip_rotation(image, a, -7.0 * i)
        result = Maneuvers.tip_projection(result, a, a.nasal_height * 0.05 * i)
        return result

    # ---------- نوک ----------
    @staticmethod
    def upturned_tip(image, landmarks, shape, intensity):
        """نوک بالا: چرخش خالص حول محور پل + کولوملا همراه — بدون تخریب."""
        i = clamp_intensity('upturned_tip', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        # سقف ۱۴° داخل محدوده طبیعی tip rotation (۱۰۰–۱۲۰°)
        result = Maneuvers.tip_rotation(image, a, 22.0 * i)
        # کولوملا نیمه همراه تا nasolabial طبیعی بماند
        result = Maneuvers.columella_set(result, a, -a.nasal_height * 0.07 * i)
        return result

    @staticmethod
    def droopy_tip(image, landmarks, shape, intensity):
        """نوک افتاده (برای متقارن‌سازی درخواست معکوس)."""
        i = clamp_intensity('droopy_tip', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        return Maneuvers.tip_rotation(image, a, -10.0 * i)

    @staticmethod
    def doll_tip(image, landmarks, shape, intensity):
        """عروسکی: نوک کوچک گرد + چرخش بالا + بال جمع."""
        i = clamp_intensity('doll_tip', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.tip_projection(image, a, -a.nasal_height * 0.05 * i)
        result = Maneuvers.tip_rotation(result, a, 17.0 * i)
        result = Maneuvers.alar_narrowing(result, a, a.nasal_width * 0.20 * i)
        return result

    # ---------- فرم‌ها ----------
    @staticmethod
    def fleshy(image, landmarks, shape, intensity):
        """گوشتی: بال پهن + نوک پر + دیواره پر."""
        i = clamp_intensity('fleshy', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.alar_flaring(image, a, a.nasal_width * 0.22 * i)
        result = Maneuvers.tip_projection(result, a, a.nasal_height * 0.06 * i)
        result = Maneuvers.sidewall_definition(result, a, a.nasal_width * 0.08 * i)
        return result

    @staticmethod
    def bony(image, landmarks, shape, intensity):
        """استخوانی: پل باریک تیز + نوک ثابت + شارپن قوس."""
        i = clamp_intensity('bony', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        return Maneuvers.dorsum_reshape(image, a, -0.16 * i)

    @staticmethod
    def fantasy(image, landmarks, shape, intensity):
        """فانتزی: باریک + نوک بالا + قوس صاف — ترکیب اروپایی کنترل‌شده."""
        i = clamp_intensity('fantasy', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.alar_narrowing(image, a, a.nasal_width * 0.30 * i)
        result = Maneuvers.dorsum_reshape(result, a, -0.12 * i)
        result = Maneuvers.tip_rotation(result, a, 18.0 * i)
        return result

    @staticmethod
    def half_fantasy(image, landmarks, shape, intensity):
        return NoseAnatomyStyles.fantasy(image, landmarks, shape,
                                         float(intensity) * 0.5)

    @staticmethod
    def natural(image, landmarks, shape, intensity):
        i = clamp_intensity('natural', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        return Maneuvers.alar_narrowing(image, a, a.nasal_width * 0.10 * i)

    # ---------- ⭐ بهترین حالت — فوق‌واقعی ----------
    @staticmethod
    def ideal_realistic(image, landmarks, shape, intensity):
        """⭐ بهترین حالت بینی — فوق‌واقعی هوشمند (AI-personalized)

        موتور IdealNoseAI: تحلیل کلینیکی همین چهره → برنامه شخصی‌سازی‌شده
        → اجرا → اندازه‌گیری مجدد روی خروجی و اصلاح پسماند (حلقه بازخورد).
        اگر موتور AI در دسترس نبود، به فرمول ثابت متعادل برمی‌گردد.
        """
        i = clamp_intensity('ideal_realistic', intensity)
        a = _resolve(landmarks, shape)
        if a is None:
            return image
        try:
            from .ideal_nose_ai import IDEAL_NOSE_AI
            result = IDEAL_NOSE_AI.transform(image, landmarks, shape, i,
                                             detector=_ai_detector())
            _rep = IDEAL_NOSE_AI.last_report
            if _rep and _rep.get('status') == 'ok':
                return result
        except Exception as e:
            logger.warning(f"ideal_nose_ai failed, using static fallback: {e}")
        # ---------- fallback ثابت (بدون AI) ----------
        result = image
        # ۱) عرض: حداکثر ۲۲٪ جمع‌کردن بال‌ها
        result = Maneuvers.alar_narrowing(result, a, a.nasal_width * 0.22 * i)
        # ۲) قوس: صاف‌سازی خیلی ملایم خط مرکزی
        result = Maneuvers.dorsum_reshape(result, a, -0.07 * i)
        # ۳) نوک: چرخش ملایم بالا + کولوملا نیمه‌همراه
        result = Maneuvers.tip_rotation(result, a, 7.0 * i)
        result = Maneuvers.columella_set(result, a, -a.nasal_height * 0.03 * i)
        # ۴) تعریف ظریف دیواره‌ها
        result = Maneuvers.sidewall_definition(result, a, a.nasal_width * 0.05 * i)
        return result

    # ---------- پل / فیلر ----------
    @staticmethod
    def filler(image, landmarks, shape, intensity):
        """فیلر: پر کردن رادیکس + برجستگی ملایم قوس."""
        i = clamp_intensity('filler', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.radix_fill(image, a, a.nasal_height * 0.08 * i)
        result = Maneuvers.dorsum_reshape(result, a, 0.05 * i)
        return result

    @staticmethod
    def slim_bridge(image, landmarks, shape, intensity):
        i = clamp_intensity('slim_bridge', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        result = Maneuvers.dorsum_reshape(image, a, -0.14 * i)
        result = Maneuvers.tip_rotation(result, a, 4.0 * i)
        return result

    @staticmethod
    def hump_reduction(image, landmarks, shape, intensity):
        """✅ قوز بینی — درخواست خیلی رایج: فرو نشاندن قوس dorsum."""
        i = clamp_intensity('hump_reduction', intensity)
        a = _resolve(landmarks, shape)
        if a is None: return image
        # قوس به سمت داخل (باریک + عقب) فقط در mid_bridge
        result = Maneuvers.dorsum_reshape(image, a, -0.18 * i)
        result = Maneuvers.tip_projection(result, a, a.nasal_height * 0.05 * i)
        return result


# ============================================
#   آداپتور سازگاری — امضای قدیمی (image, points, intensity)
# ============================================

class NoseStyles:
    """
    اگر points از face_parser.get_points آمده باشد (۱۸ نقطه)، آناتومی کامل
    در دسترس نیست؛ ولی چون beauty_engine حالا لندمارک کامل را دارد،
    این آداپتور فقط برای سازگاری قدیمی نگه داشته شده.
    """

    _MAP = {
        'smaller': NoseAnatomyStyles.smaller, 'bigger': NoseAnatomyStyles.bigger,
        'narrower': NoseAnatomyStyles.narrower, 'wider': NoseAnatomyStyles.wider,
        'shorter': NoseAnatomyStyles.shorter, 'longer': NoseAnatomyStyles.longer,
        'upturned_tip': NoseAnatomyStyles.upturned_tip,
        'doll_tip': NoseAnatomyStyles.doll_tip,
        'fleshy': NoseAnatomyStyles.fleshy, 'bony': NoseAnatomyStyles.bony,
        'fantasy': NoseAnatomyStyles.fantasy,
        'half_fantasy': NoseAnatomyStyles.half_fantasy,
        'natural': NoseAnatomyStyles.natural,
        'ideal_realistic': NoseAnatomyStyles.ideal_realistic,
        'filler': NoseAnatomyStyles.filler,
        'slim_bridge': NoseAnatomyStyles.slim_bridge,
        'hump_reduction': NoseAnatomyStyles.hump_reduction,
    }

    @staticmethod
    def apply(image, action, landmarks, shape, intensity):
        fn = NoseStyles._MAP.get(action)
        if fn is None:
            return image
        return fn(image, landmarks, shape, intensity)

    # سازگاری امضای قدیمی (points) — از polygon تخمین می‌زند
    @staticmethod
    def _legacy(image, points, intensity, action):
        a = NoseAnatomy(fallback_polygon=points)
        if not a.valid:
            return image
        fn = NoseStyles._MAP.get(action)
        return fn(image, a.ordered_array().tolist(), (image.shape, ), intensity) \
            if fn else image


class NoseWarping:
    smaller = staticmethod(NoseAnatomyStyles.smaller)
    bigger = staticmethod(NoseAnatomyStyles.bigger)
    ideal_realistic = staticmethod(NoseAnatomyStyles.ideal_realistic)
    narrower = staticmethod(NoseAnatomyStyles.narrower)
    wider = staticmethod(NoseAnatomyStyles.wider)
    shorter = staticmethod(NoseAnatomyStyles.shorter)
    longer = staticmethod(NoseAnatomyStyles.longer)
    upturned_tip = staticmethod(NoseAnatomyStyles.upturned_tip)
    doll_tip = staticmethod(NoseAnatomyStyles.doll_tip)
    fleshy = staticmethod(NoseAnatomyStyles.fleshy)
    bony = staticmethod(NoseAnatomyStyles.bony)
    fantasy = staticmethod(NoseAnatomyStyles.fantasy)
    half_fantasy = staticmethod(NoseAnatomyStyles.half_fantasy)
    natural = staticmethod(NoseAnatomyStyles.natural)
    filler = staticmethod(NoseAnatomyStyles.filler)
    slim_bridge = staticmethod(NoseAnatomyStyles.slim_bridge)
    hump_reduction = staticmethod(NoseAnatomyStyles.hump_reduction)
