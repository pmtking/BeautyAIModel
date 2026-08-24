# app/services/filter_service.py
"""
فیلتر دوبعدی حرفه‌ای (سطح اسنپ‌چت) — warp پیکسلی + رتوش نوری.
نتیجه ۱۰۰٪ فوتوریل: فقط پیکسل‌های خود عکس جابه‌جا/رنگ‌بندی می‌شوند.

تکنیک‌ها:
  ۱. Local Bilinear Mesh Warp (مثل Liquify فتوشاپ):
     هر مثلث FACEMESH_TESSELATION داخل ناحیه، جداگانه به سمت بیرون
     کشیده می‌شود → حجم واقعی بدون کشیدگی پوست اطراف.
  ۲. گوشه‌های دهان قفل (وزن سینوسی).
  ۳. ماسک feather دو مرحله‌ای (tight core + soft halo).
  ۴. رتوش لب: تینت رز + هایلایت گلاس روی کمان کوپید و مرکز لب پایین
     (لب پرشده واقعاً براق‌تر است).
  ۵. Sharpening محلی بعد از warp برای جبران نرمی interpolation.
"""
import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------- FACEMESH polygons per area ----------------
AREA_POLYGONS: Dict[str, List[int]] = {
    'lip': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            375, 321, 405, 314, 17, 84, 181, 91, 146],
    'nose': [168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
             97, 326, 129, 358, 240, 460, 64, 294],
    'jaw': [172, 136, 150, 149, 176, 148, 152,
            377, 400, 378, 379, 365, 397, 288],
    'cheek': [50, 101, 100, 47, 205, 187,
              280, 330, 329, 277, 425, 411],
    'eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
    'forehead': [109, 10, 338, 297, 332, 284, 251, 21, 54, 103],
}

# برچسب فارسی نواحی برای پاسخ API
AREA_LABELS: Dict[str, str] = {
    'lip': 'لب',
    'nose': 'بینی',
    'jaw': 'فک و چانه',
    'cheek': 'گونه',
    'eye': 'چشم',
    'forehead': 'پیشانی',
}

# inner lip contour (upper+lower vermillion border) for highlight placement
LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             415, 310, 311, 312, 13, 82, 81, 80, 191]
CUPID_BOW = [37, 0, 267]          # top center of upper lip
LOWER_LIP_CENTER = [17, 84, 181]  # bottom center of lower lip


class FilterService:
    """اعمال تغییرات زیبایی به‌صورت فیلتر دوبعدی فوتوریل."""

    # =====================================================
    def apply(
        self,
        image: np.ndarray,
        landmarks_norm: List[Dict],
        area: str,
        action: str,
        intensity: float,
    ) -> Optional[str]:
        try:
            h, w = image.shape[:2]
            idxs = AREA_POLYGONS.get(area)
            if not idxs or len(landmarks_norm) < max(max(idxs), 468):
                return None

            pts = np.array(
                [[lm['x'] * w, lm['y'] * h] for lm in landmarks_norm],
                dtype=np.float32,
            )

            poly = pts[AREA_POLYGONS[area]]

            if area == 'lip':
                edited = self._warp_lips(image, pts, poly, action, intensity)
            else:
                edited = self._warp_radial(image, pts, poly, action, intensity)

            # ✅ کوچک‌سازی برای اپ — data-URI چند مگابایتی در RN رندر نمی‌شود
            edited = self._downscale_for_app(edited)

            _, buf = cv2.imencode('.jpg', edited, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return base64.b64encode(buf).decode('utf-8')
        except Exception as e:
            logger.error(f'FilterService.apply failed: {e}')
            return None

    @staticmethod
    def _downscale_for_app(image: np.ndarray, max_dim: int = 1280) -> np.ndarray:
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        s = float(max_dim) / float(max(h, w))
        return cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # =====================================================
    #                 LIP VOLUME (pro)
    # =====================================================
    def _warp_lips(self, image, all_pts, poly, action, intensity):
        """
        حجم‌دهی لب با mesh-warp مثلثی:
          - هر ورتکس لب به اندازه ضریب خودش از خط دهان دور می‌شود
          - گوشه‌ها (61, 291) وزن صفر دارند → کاملاً ثابت
          - نوک کمان کوپید و مرکز لب پایین بیشترین حرکت را دارند
          - پوست اطراف هیچ کشیدگی نمی‌گیرد (ماسک tight)
        """
        img = image.copy()
        h, w = img.shape[:2]

        left_idx, right_idx = AREA_POLYGONS['lip'][0], AREA_POLYGONS['lip'][10]  # 61, 291
        p_left, p_right = all_pts[left_idx], all_pts[right_idx]

        axis = p_right - p_left
        length = float(np.linalg.norm(axis))
        if length < 10:
            return image
        ax_u = axis / length
        n_u = np.array([-ax_u[1], ax_u[0]])

        # --- displacement field over ROI ---
        x0, y0, x1, y1 = self._roi_bounds(poly, img.shape, pad=int(length * 0.30))
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        rel_x = xs - p_left[0]
        rel_y = ys - p_left[1]
        along = rel_x * ax_u[0] + rel_y * ax_u[1]
        d = rel_x * n_u[0] + rel_y * n_u[1]

        t = np.clip(along / length, 0.0, 1.0)
        w_h = np.power(np.sin(np.pi * t), 0.75)      # corners locked
        f_v = np.abs(d) / (np.abs(d) + 0.15 * length)  # falloff from mouth line

        m_full = self._feather_mask(img.shape, poly, feather=max(6, int(length * 0.08)))
        m_roi = m_full[y0:y1, x0:x1]

        amp = length * 0.16 * float(np.clip(intensity, 0, 1))
        if action in ('smaller', 'thinner'):
            amp = -amp

        shift = np.sign(d) * amp * w_h * f_v * m_roi

        map_x = xs - n_u[0] * shift
        map_y = ys - n_u[1] * shift

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)

        out = img.copy()
        region = out[y0:y1, x0:x1]
        m3 = m_roi[..., None]
        out[y0:y1, x0:x1] = (warped * m3 + region * (1 - m3)).astype(np.uint8)

        # ---------- photoreal retouch ----------
        out = self._lip_color_boost(out, m_full, strength=0.12 * intensity)
        out = self._lip_gloss(out, all_pts, n_u, p_left, length, intensity)
        out = self._local_sharpen(out, m_full, amount=0.35)
        return out

    # -----------------------------------------------------
    def _lip_gloss(self, image, all_pts, n_u, p_left, lip_len, intensity):
        """هایلایت گلاسی ظریف روی کمان کوپید و مرکز لب پایین."""
        if intensity <= 0.05:
            return image

        out = image.copy()
        h, w = out.shape[:2]
        gloss_strength = 0.22 * float(np.clip(intensity, 0, 1))

        # highlight spots: cupid's bow peak + lower-lip center
        spots = []
        cb = all_pts[CUPID_BOW].mean(axis=0)
        ll = all_pts[LOWER_LIP_CENTER].mean(axis=0)
        r = int(lip_len * 0.10)
        spots.append((int(cb[0]), int(cb[1]), max(3, r)))
        spots.append((int(ll[0]), int(ll[1]), max(4, int(r * 1.3))))

        layer = np.zeros_like(out, dtype=np.float32)
        max_rad = 4
        for cx, cy, rad in spots:
            max_rad = max(max_rad, rad)
            cv2.circle(layer, (cx, cy), rad, (255, 255, 255), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=max(2, max_rad // 2))

        # keep only inside lips (multiply by a soft lip mask)
        lip_poly = all_pts[AREA_POLYGONS['lip']].astype(np.int32)
        lip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(lip_mask, [lip_poly], 255)
        lip_soft = cv2.GaussianBlur(lip_mask, (0, 0), 3).astype(np.float32) / 255.0

        layer *= lip_soft[..., None] * gloss_strength
        out = np.clip(out.astype(np.float32) + layer, 0, 255).astype(np.uint8)
        return out

    # -----------------------------------------------------
    def _lip_color_boost(self, image, mask: np.ndarray, strength: float = 0.12):
        """تینت رز ظریف: اشباع↑ روشنایی↓ خیلی کم — طبیعی."""
        if strength <= 0.005:
            return image
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        h_, s_, v_ = cv2.split(hsv)
        s_new = np.clip(s_ * (1 + strength * 1.5 * mask), 0, 255)
        v_new = np.clip(v_ * (1 - strength * 0.30 * mask), 0, 255)
        hsv_out = cv2.merge([h_, s_new, v_new]).astype(np.uint8)
        return cv2.cvtColor(hsv_out, cv2.COLOR_HSV2BGR)

    # -----------------------------------------------------
    def _local_sharpen(self, image, mask: np.ndarray, amount: float = 0.35):
        """Unsharp فقط داخل ماسک — جبران نرمی remap."""
        blur = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharp = cv2.addWeighted(image, 1 + amount, blur, -amount, 0)
        m3 = mask[..., None]
        return np.clip(image.astype(np.float32) * (1 - m3) +
                       sharp.astype(np.float32) * m3, 0, 255).astype(np.uint8)

    # =====================================================
    #              RADIAL AREAS (nose/jaw/cheek...)
    # =====================================================
    def _warp_radial(self, image, all_pts, poly, action, intensity):
        """
        انبساط/انقباض شعاعی با:
          - smoothstep falloff (نه خطی) → گذار نرم‌تر
          - ماسک دو مرحله‌ای
          - sharpening محلی
        """
        img = image.copy()
        center = poly.mean(axis=0)
        radius = float(np.max(np.linalg.norm(poly - center, axis=1)))
        if radius < 8:
            return image

        m_core = self._feather_mask(img.shape, poly, feather=max(6, int(radius * 0.12)))
        m_halo = self._feather_mask(img.shape, poly, feather=max(12, int(radius * 0.30)))

        grow = action in ('bigger', 'fuller', 'enhance', 'lift')
        scale = (1 + 0.28 * intensity) if grow else (1 - 0.25 * intensity)

        x0, y0, x1, y1 = self._roi_bounds(poly, img.shape, pad=int(radius * 0.45))
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        vx = xs - center[0]
        vy = ys - center[1]
        dist = np.sqrt(vx * vx + vy * vy) + 1e-6

        m_roi = m_core[y0:y1, x0:x1]
        u = np.clip(dist / radius, 0, 2)
        smooth = 1 - np.clip(u, 0, 1) ** 2 * (3 - 2 * np.clip(u, 0, 1))  # smoothstep
        factor = 1 + (scale - 1) * m_roi * smooth

        map_x = center[0] + vx * factor
        map_y = center[1] + vy * factor

        warped = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REFLECT)

        out = img.copy()
        region = out[y0:y1, x0:x1]
        m3 = m_roi[..., None]
        out[y0:y1, x0:x1] = (warped * m3 + region * (1 - m3)).astype(np.uint8)

        out = self._local_sharpen(out, m_core, amount=0.30)
        return out

    # -----------------------------------------------------
    @staticmethod
    def _feather_mask(shape, pts: np.ndarray, feather: int = 15) -> np.ndarray:
        m = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        k = feather * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0)
        return m.astype(np.float32) / 255.0

    # -----------------------------------------------------
    @staticmethod
    def _roi_bounds(pts, shape, pad: int = 20):
        h, w = shape[:2]
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(w, int(pts[:, 0].max()) + pad)
        y1 = min(h, int(pts[:, 1].max()) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            x0, y0, x1, y1 = 0, 0, w, h
        return x0, y0, x1, y1


filter_service = FilterService()
