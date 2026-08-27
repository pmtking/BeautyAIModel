"""
🚀 NarrowingPro — موتور باریک‌سازی حرفه‌ای بینی (نسخه نهایی ۲۴ اوت)
====================================================================
کشف‌های diag1-16:
  1. علامت dx در _warp_field برای «جمع به داخل» برعکس بود → اصلاح شد
  2. warp بافت کافی نیست؛ MediaPipe/چشم لبه را از «سایه» میخواند
  3. Shading Transfer ضروری است: سایه قدیمی روشن + لبه جدید تیره
  4. معیار سنجش: خط سایه آلار (پیکسلی) — نه لندمارک MP که ±15px خطا دارد

نتیجه تست: frac=0.2 → 17.6% کاهش واقعی سایه (88% محقق‌شده نسبت به هدف)
"""
import cv2
import numpy as np
from typing import Optional

try:
    from .nose_anatomy import NoseAnatomy
    from .nose_styles import _roi_bounds
except Exception:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.dirname(__file__))
    from nose_anatomy import NoseAnatomy
    from nose_styles import _roi_bounds


class NarrowingPro:
    """باریک‌سازی بال‌ها با warp + shading transfer — سطح جراحی"""

    @staticmethod
    def apply(image: np.ndarray, anat: NoseAnatomy,
              target_frac: float = 0.25,
              shade_strength: float = 24.0) -> np.ndarray:
        """
        target_frac: نسبت کاهش عرض هدف — 0.20 یعنی ۲۰٪ باریک‌تر
        shade_strength: قدرت shading (18 ملایم / 24 پیش‌فرض / 30 قوی)
        """
        if anat is None or not anat.valid:
            return image

        w = anat.nasal_width
        h = anat.nasal_height
        al = anat.get('alar_l')
        ar = anat.get('alar_r')
        if al is None or ar is None:
            return image

        frac = float(np.clip(target_frac, 0.03, 0.45))
        shift_px = w * frac
        reach = w * 1.15
        pts = anat.ordered_array()
        if pts is None:
            return image
        radix = anat.get('radix')
        if radix is None:
            return image
        x0, y0, x1, y1 = _roi_bounds(image.shape, pts, pad_frac=0.75)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        # ═══════════ ۱) Warp دوطرفه با علامت صحیح ═══════════
        # ⚠️ semantics واقعی remap: out(x)=img(x+d) → محتوای x در x+d ظاهر
        # میشود؛ یعنی برای جمع‌کردن آلار چپ به داخل (به راست)، باید از چپ
        # نمونه برداریم: منفی. (تأییدشده با template-matching — T-101)
        win_l = np.clip(1 - np.abs(xs - al[0]) / reach, 0, 1) ** 1.25
        win_r = np.clip(1 - np.abs(xs - ar[0]) / reach, 0, 1) ** 1.25

        roi = image[y0:y1, x0:x1]
        map_x = ((xs - x0) - win_l * shift_px + win_r * shift_px).astype(np.float32)
        map_y = (ys - y0).astype(np.float32)
        warped = cv2.remap(roi, map_x, map_y,
                           cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        m = np.maximum(win_l, win_r)[..., None] * 0.97
        out_roi = (warped * m + roi * (1 - m)).astype(np.uint8)

        out = image.copy()
        out[y0:y1, x0:x1] = out_roi

        # ═══════════ ۲) Shading Transfer ═══════════
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[..., 0]
        sigma = w * 0.055

        g_old = (np.exp(-((xs - al[0])**2) / (2*sigma**2)) +
                 np.exp(-((xs - ar[0])**2) / (2*sigma**2)))
        new_al = al[0] + shift_px * 0.9
        new_ar = ar[0] - shift_px * 0.9
        g_new = (np.exp(-((xs - new_al)**2) / (2*sigma**2)) +
                 np.exp(-((xs - new_ar)**2) / (2*sigma**2)))

        delta = shade_strength * g_old - shade_strength * 0.8 * g_new

        radix_y = radix[1]
        y_alar = int((al[1] + ar[1]) / 2)
        band = ((ys > radix_y - w * 0.08) &
                (ys < y_alar + h * 0.30)).astype(np.float32)

        L[y0:y1, x0:x1] = np.clip(
            L[y0:y1, x0:x1] + delta * band, 0, 255)
        lab[..., 0] = L
        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # ═══════════ ۳) تقویت ظریف لبه جدید ═══════════
        edge = np.zeros((y1-y0, x1-x0), np.float32)
        for ex in (new_al, new_ar):
            d = np.abs(xs - ex)
            edge += np.clip(1 - d/(w*0.05), 0, 1).astype(np.float32)
        edge = np.clip(edge, 0, 1)[..., None] * band[..., None]
        roi_blur = cv2.GaussianBlur(out[y0:y1, x0:x1], (0, 0), 2.0)
        sharp_roi = cv2.addWeighted(out[y0:y1, x0:x1], 1.15, roi_blur, -0.15, 0)
        out[y0:y1, x0:x1] = (sharp_roi*edge + out[y0:y1, x0:x1]*(1-edge)).astype(np.uint8)

        return out

    # ---------------------------------------------------------
    @staticmethod
    def measure_shade_width(image: np.ndarray, anat: NoseAnatomy) -> float:
        """اندازه‌گیری مستقل عرض بینی از روی خط سایه آلار (پیکسلی)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        al, ar = anat.get('alar_l'), anat.get('alar_r')
        if al is None or ar is None:
            return float(anat.nasal_width)
        w = anat.nasal_width
        y_c = int((al[1]+ar[1])/2)
        widths = []
        for dy in (-12, -6, 0, 6, 12):
            row = gray[y_c+dy]
            win = int(w*0.30)
            seg_l = row[max(0, int(al[0])-win):int(al[0])+int(w*0.15)]
            seg_r = row[int(ar[0])-int(w*0.15):int(ar[0])+win]
            if len(seg_l) < 5 or len(seg_r) < 5:
                continue
            xl = max(0, int(al[0])-win) + int(np.argmin(seg_l))
            xr = int(ar[0])-int(w*0.15) + int(np.argmin(seg_r))
            widths.append(xr-xl)
        return float(np.median(widths)) if widths else float(w)


# سازگاری با امضای قدیمی Maneuvers.alar_narrowing(image, anat, amount_px)
def alar_narrowing_fixed(image: np.ndarray, anat: NoseAnatomy, amount_px: float):
    """آداپتور: amount_px قدیمی → frac جدید"""
    if anat is None or not anat.valid:
        return image
    frac = float(np.clip(amount_px / max(anat.nasal_width, 1e-6), 0.03, 0.45))
    return NarrowingPro.apply(image, anat, frac)
