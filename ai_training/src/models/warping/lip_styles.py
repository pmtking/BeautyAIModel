"""
Lip Styles — انواع لب‌های درخواستی
===================================
استایل‌ها:
  russian        روسی            حجم بالا + کمان کوپید مشخص + گوشه بالا
  brazilian      برزیلی          حجم بالا + گوشه‌های رو به بالا
  hollywood      هالیوودی        حجم زیاد + برجستگی مرکزی
  heart_shape    قلوه‌ای          فرم قلب با کمان کوپید عمیق
  classic        کلاسیک          متعادل و طبیعی
  natural        طبیعی            حجم ملایم روزمره
  fuller         پرتر (پایه)      حجم یکنواخت بالا و پایین
  thinner        باریک‌تر
  cupids_bow     کمان کوپید تیز   فقط تعریف کمان کوپید
  corner_lift    گوشه‌ها بالا     لیفت گوشه لب (لبخند)
"""
import cv2
import numpy as np
from typing import List, Optional, Tuple

logger = __import__('logging').getLogger(__name__)


# ============================================
#   هندسهٔ لب
# ============================================

class LipGeo:
    """استخراج لنگرهای لب از پلی‌گان ۲۰ نقطه‌ای استاندارد (یا fallback)."""

    def __init__(self, points: List[List[int]]):
        pts = np.array(points, dtype=np.float32)
        self.pts = pts
        n = len(pts)

        if n == 20:
            # ترتیب: [61,185,40,39,37,0,267,269,270,409,291,375,321,405,314,17,84,181,91,146]
            self.left = pts[0]        # گوشه چپ (MP61)
            self.right = pts[10]      # گوشه راست (MP291)
            self.cupid = pts[5]       # نوک کمان کوپید (MP0)
            self.lower = pts[15]      # مرکز لب پایین (MP17)
            self.precise = True
        else:
            self.left = pts[int(np.argmin(pts[:, 0]))]
            self.right = pts[int(np.argmax(pts[:, 0]))]
            self.cupid = pts[int(np.argmin(pts[:, 1]))]
            self.lower = pts[int(np.argmax(pts[:, 1]))]
            self.precise = False

        self.center = pts.mean(axis=0)
        axis = self.right - self.left
        self.length = float(np.linalg.norm(axis))
        self.length = max(self.length, 10.0)
        self.ax_u = axis / self.length          # واحد در راستای لب
        self.n_u = np.array([-self.ax_u[1], self.ax_u[0]])  # عمود بر لب

    def rel(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """مختصات (along, d) یک نقطه در دستگاه مختصات لب."""
        rel = p - self.left
        return (rel @ self.ax_u, rel @ self.n_u)


# ============================================
#   عملیات warp
# ============================================

def _roi_bounds(shape, pts: np.ndarray, pad_frac: float = 0.45):
    h, w = shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = max(x1 - x0, y1 - y0)
    pad = int(r * pad_frac)
    rx0, ry0 = int(max(0, cx - r / 2 - pad)), int(max(0, cy - r / 2 - pad))
    rx1, ry1 = int(min(w, cx + r / 2 + pad)), int(min(h, cy + r / 2 + pad))
    if rx1 - rx0 < 4 or ry1 - ry0 < 4:
        return 0, 0, w, h
    return rx0, ry0, rx1, ry1


def _feather_mask(shape, poly: np.ndarray, feather: int = 10) -> np.ndarray:
    m = np.zeros(shape[:2], dtype=np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
    k = max(3, feather * 2 + 1)
    m = cv2.GaussianBlur(m, (k, k), 0)
    return m.astype(np.float32) / 255.0


class LipWarpOps:
    """عملیات warp مخصوص لب — همه در دستگاه مختصات لب (along, d)."""

    @staticmethod
    def volume(image: np.ndarray, geo: LipGeo, poly: np.ndarray,
               upper_scale: float, lower_scale: float,
               corner_lock: float = 1.0) -> np.ndarray:
        """
        حجم‌دهی تفکیکی: لب بالا و پایین هرکدام ضریب خودشان را دارند.
        گوشه‌ها با وزن سینوسی قفل می‌شوند.
        """
        img = image.copy()
        h, w = img.shape[:2]

        x0, y0, x1, y1 = _roi_bounds(shape=img.shape, pts=geo.pts)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        rel_x = xs - geo.left[0]
        rel_y = ys - geo.left[1]
        along = rel_x * geo.ax_u[0] + rel_y * geo.ax_u[1]
        d = rel_x * geo.n_u[0] + rel_y * geo.n_u[1]

        t = np.clip(along / geo.length, 0.0, 1.0)
        w_h = np.power(np.sin(np.pi * t), 0.75) ** corner_lock  # گوشه‌ها ثابت

        m_full = _feather_mask(img.shape, poly, feather=max(5, int(geo.length * 0.07)))
        m_roi = m_full[y0:y1, x0:x1]

        upper = (d < 0)
        f_v_u = np.abs(d) / (np.abs(d) + 0.18 * geo.length)
        shift = np.where(
            upper,
            upper_scale * w_h * f_v_u,
            lower_scale * w_h * f_v_u,
        ) * m_roi

        map_x = xs - geo.n_u[0] * shift
        map_y = ys - geo.n_u[1] * shift

        warped = cv2.remap(img[y0:y1, x0:x1], map_x - x0, map_y - y0,
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        out = img.copy()
        m3 = m_roi[..., None]
        out[y0:y1, x0:x1] = (warped * m3 + out[y0:y1, x0:x1] * (1 - m3)).astype(np.uint8)
        return out

    @staticmethod
    def move_point(image: np.ndarray, geo: LipGeo, anchor: np.ndarray,
                   radius: float, dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
        """جابه‌جایی نرم حول یک نقطه (مثلاً بالا کشیدن گوشه لب)."""
        img = image.copy()
        h, w = img.shape[:2]
        if radius <= 1 or (dx == 0 and dy == 0):
            return img

        x0, y0, x1, y1 = _roi_bounds(shape=img.shape, pts=geo.pts)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        dist = np.sqrt((xs - anchor[0]) ** 2 + (ys - anchor[1]) ** 2)
        u = np.clip(dist / max(radius, 1e-6), 0, 1)
        win = 0.5 * (1 + np.cos(np.pi * u))

        map_x = xs - dx * win
        map_y = ys - dy * win

        warped = cv2.remap(img[y0:y1, x0:x1], map_x - x0, map_y - y0,
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        out = img.copy()
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        mask[win > 0.02] = win[win > 0.02]
        m3 = mask[..., None]
        out[y0:y1, x0:x1] = (warped * m3 + out[y0:y1, x0:x1] * (1 - m3)).astype(np.uint8)
        return out


# ============================================
#   استایل‌ها
# ============================================

class LipStyles:
    """هر متد: (image, points, intensity) → image ویرایش‌شده."""

    @staticmethod
    def _vol(image, points, intensity, up: float, low: float,
             corner_lock: float = 1.0):
        geo = LipGeo(points)
        poly = geo.pts
        amp = geo.length * 0.22 * float(np.clip(intensity, 0, 1))
        return LipWarpOps.volume(image, geo, poly,
                                 upper_scale=up * amp, lower_scale=low * amp,
                                 corner_lock=corner_lock)

    # ---------- پایه ----------
    @staticmethod
    def fuller(image, points, intensity):
        """پرتر: حجم یکنواخت بالا و پایین."""
        return LipStyles._vol(image, points, intensity, up=1.0, low=1.15)

    @staticmethod
    def thinner(image, points, intensity):
        """باریک‌تر: جمع کردن لب به سمت خط دهان."""
        return LipStyles._vol(image, points, intensity, up=-0.8, low=-0.9)

    @staticmethod
    def natural(image, points, intensity):
        """طبیعی: حجم ملایم روزمره."""
        return LipStyles._vol(image, points, intensity, up=0.6, low=0.8)

    @staticmethod
    def classic(image, points, intensity):
        """کلاسیک: متعادل، گوشه‌ها کمی بالا."""
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity, up=0.8, low=1.0)
        lift = geo.length * 0.06 * intensity
        result = LipWarpOps.move_point(result, geo, geo.left, geo.length * 0.22, dy=-lift)
        result = LipWarpOps.move_point(result, geo, geo.right, geo.length * 0.22, dy=-lift)
        return result

    # ---------- استایل‌های محبوب ----------
    @staticmethod
    def russian(image, points, intensity):
        """
        روسی: حجم بالا + کمان کوپید خیلی مشخص + مرکز لب بالا پرتر.
        ویژگی اصلی لب روسی: لب بالا بلندتر از حالت عادی دیده می‌شود.
        """
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity, up=1.25, low=0.95,
                                corner_lock=1.2)
        # برجسته‌تر کردن نوک کمان کوپید
        lift = geo.length * 0.10 * intensity
        result = LipWarpOps.move_point(result, geo, geo.cupid,
                                       geo.length * 0.28, dy=-lift)
        return result

    @staticmethod
    def brazilian(image, points, intensity):
        """برزیلی: حجم بالا + گوشه‌های رو به بالا (لبخند ملایم)."""
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity, up=1.0, low=1.2)
        lift = geo.length * 0.09 * intensity
        result = LipWarpOps.move_point(result, geo, geo.left, geo.length * 0.25, dy=-lift)
        result = LipWarpOps.move_point(result, geo, geo.right, geo.length * 0.25, dy=-lift)
        return result

    @staticmethod
    def hollywood(image, points, intensity):
        """هالیوودی: حجم زیاد هر دو لب + برجستگی مرکز لب پایین."""
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity, up=1.15, low=1.35)
        out = geo.length * 0.08 * intensity
        result = LipWarpOps.move_point(result, geo, geo.lower,
                                       geo.length * 0.30, dy=out)
        return result

    @staticmethod
    def heart_shape(image, points, intensity):
        """قلوه‌ای: کمان کوپید عمیق + مرکز لب بالا فرورفته → فرم قلب."""
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity, up=0.9, low=1.15)
        lift = geo.length * 0.12 * intensity
        result = LipWarpOps.move_point(result, geo, geo.cupid,
                                       geo.length * 0.25, dy=-lift)
        return result

    @staticmethod
    def cupids_bow(image, points, intensity):
        """کمان کوپید تیز: فقط تعریف کمان کوپید بدون حجم زیاد."""
        geo = LipGeo(points)
        result = LipStyles._vol(image, points, intensity * 0.4, up=0.7, low=0.5)
        lift = geo.length * 0.14 * intensity
        result = LipWarpOps.move_point(result, geo, geo.cupid,
                                       geo.length * 0.22, dy=-lift)
        return result

    @staticmethod
    def corner_lift(image, points, intensity):
        """لیفت گوشه لب: گوشه‌ها به بالا — لبخند بدون جراحی."""
        geo = LipGeo(points)
        lift = geo.length * 0.14 * intensity
        result = LipWarpOps.move_point(image, geo, geo.left, geo.length * 0.30, dy=-lift)
        result = LipWarpOps.move_point(result, geo, geo.right, geo.length * 0.30, dy=-lift)
        return result


# ============================================
#   سازگاری با کد قدیمی (LipWarping)
# ============================================

class LipWarping:
    fuller = staticmethod(LipStyles.fuller)
    thinner = staticmethod(LipStyles.thinner)
    natural = staticmethod(LipStyles.natural)
    russian = staticmethod(LipStyles.russian)
    heart_shape = staticmethod(LipStyles.heart_shape)
    brazilian = staticmethod(LipStyles.brazilian)
    hollywood = staticmethod(LipStyles.hollywood)
    classic = staticmethod(LipStyles.classic)
    cupids_bow = staticmethod(LipStyles.cupids_bow)
    corner_lift = staticmethod(LipStyles.corner_lift)
