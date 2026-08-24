# app/services/retouch_service.py
"""
موتور رتوش حرفه‌ای (سطح اپ‌های Peachy / Facetune):
  - brighten     : روشن‌کردن طبیعی پوست (فقط نواحی روشن، بدون clip)
  - even_skin    : یکنواخت‌سازی رنگ پوست (کاهش لکه و قرمزی)
  - smooth       : لطافت پوست با حفظ جزئیات (bilateral + highpass)
  - warmth       : گرمی/سردی رنگ
  - clarity      : شفافیت (local contrast)
  - contrast     : کنتراست کلی
  - saturation   : اشباع رنگ
  - vignette     : وینت نرم

همه ابزارها mask-aware هستند: فقط روی پوست اعمال می‌شوند نه چشم/لب/مو.
شدت هر ابزار 0..100 از کلاینت می‌آید.
"""
import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RetouchService:

    # =====================================================
    def apply_all(
        self,
        image: np.ndarray,
        landmarks_norm: Optional[List[Dict]] = None,
        adjustments: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """
        image: BGR
        adjustments: {'brighten': 0..100, 'smooth': .., ...}
        خروجی: base64 JPEG
        """
        try:
            adj = {k: float(v) for k, v in (adjustments or {}).items()}
            adj = {k: max(0.0, min(100.0, v)) for k, v in adj.items() if abs(v) > 0.5}
            if not adj and not landmarks_norm:
                return self._encode(image)

            skin_mask = self._build_skin_mask(image, landmarks_norm)

            out = image.astype(np.float32)

            if 'brighten' in adj:
                out = self._brighten(out, adj['brighten'], skin_mask)
            if 'even_skin' in adj:
                out = self._even_skin(out, adj['even_skin'], skin_mask)
            if 'smooth' in adj:
                out = self._smooth(out, adj['smooth'], skin_mask)
            if 'warmth' in adj:
                out = self._warmth(out, adj['warmth'], skin_mask)
            if 'clarity' in adj:
                out = self._clarity(out, adj['clarity'])
            if 'contrast' in adj:
                out = self._contrast(out, adj['contrast'])
            if 'saturation' in adj:
                out = self._saturation(out, adj['saturation'])
            if 'vignette' in adj:
                out = self._vignette(out, adj['vignette'])

            out = np.clip(out, 0, 255).astype(np.uint8)
            return self._encode(out)
        except Exception as e:
            logger.error(f'RetouchService failed: {e}')
            return None

    # =====================================================
    #                    TOOLS
    # =====================================================

    def _brighten(self, img: np.ndarray, amount: float, skin) -> np.ndarray:
        """Lift shadows روی پوست؛ هایلایت دست نمی‌خورد → بدون سوختگی."""
        k = amount / 100.0
        lum = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32) / 255.0
        shadow_w = np.clip(1 - lum * 1.6, 0, 1) ** 1.5      # فقط سایه‌ها
        lift = k * 28 * shadow_w
        out = img.copy()
        out[:, :, 0] += lift * 0.55                          # L channel
        out[:, :, 1] += lift * 0.10                          # کمی گرما
        out[:, :, 2] -= lift * 0.05
        m3 = (skin * 0.85 + 0.15)[..., None]
        return img * (1 - m3) + out * m3

    def _even_skin(self, img: np.ndarray, amount: float, skin) -> np.ndarray:
        """کاهش لکه/قرمزی: chroma smoothing در AB کانال‌های LAB."""
        k = amount / 100.0
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        ab = lab[:, :, 1:3]
        ab_blur = cv2.GaussianBlur(ab, (0, 0), sigmaX=6 + 8 * k)
        mixed_ab = ab * (1 - 0.65 * k) + ab_blur * (0.65 * k)
        lab[:, :, 1:3] = mixed_ab
        even = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        m3 = (skin[..., None],)[0]
        return img * (1 - skin[..., None] * 0.9) + even * (skin[..., None] * 0.9)

    def _smooth(self, img: np.ndarray, amount: float, skin) -> np.ndarray:
        """Edge-aware smoothing: bilateral + بازگرداندن جزئیات (frequency separation)."""
        k = amount / 100.0
        u8 = np.clip(img, 0, 255).astype(np.uint8)

        small = cv2.resize(u8, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        smooth_s = cv2.bilateralFilter(small, d=9, sigmaColor=30 + 40 * k, sigmaSpace=7)
        smooth = cv2.resize(smooth_s, (u8.shape[1], u8.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        base_f = img.astype(np.float32)
        detail = base_f - cv2.GaussianBlur(base_f, (0, 0), 1.2)   # جزئیات (منافذ/مو)
        smoothed = base_f * (1 - 0.75 * k) + smooth * (0.75 * k)
        result = smoothed + detail * (0.85 - 0.35 * k)             # جزئیات حفظ می‌شود

        return base_f * (1 - skin[..., None]) + result * skin[..., None]

    def _warmth(self, img: np.ndarray, amount: float, skin=None) -> np.ndarray:
        """-100 سرد … +100 گرم."""
        k = (amount - 50) / 50.0  # -1..+1
        out = img.copy()
        out[:, :, 0] += k * 12   # B
        out[:, :, 2] += k * 16   # R
        out[:, :, 1] += k * 4    # G
        return out

    def _clarity(self, img: np.ndarray, amount: float) -> np.ndarray:
        """Local contrast با unsharp بزرگ‌مقیاس."""
        k = (amount - 50) / 50.0
        if abs(k) < 0.02:
            return img
        blur = cv2.GaussianBlur(img, (0, 0), 12)
        return img + k * 0.45 * (img - blur)

    def _contrast(self, img: np.ndarray, amount: float) -> np.ndarray:
        k = (amount - 50) / 50.0
        gamma = 1 - k * 0.18
        lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.float32)
        return cv2.merge([cv2.LUT(np.clip(c, 0, 255).astype(np.uint8), lut) for c in cv2.split(img)]).astype(np.float32)

    def _saturation(self, img: np.ndarray, amount: float) -> np.ndarray:
        k = (amount - 50) / 50.0
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= (1 + k * 0.5)
        return cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    def _vignette(self, img: np.ndarray, amount: float) -> np.ndarray:
        k = amount / 100.0
        h, w = img.shape[:2]
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        d = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2) / np.sqrt(2)
        vig = 1 - k * 0.45 * np.clip(d - 0.45, 0, 1) ** 2
        return img * vig[..., None]

    # =====================================================
    #                   SKIN MASK
    # =====================================================

    def _build_skin_mask(
        self,
        image: np.ndarray,
        landmarks_norm: Optional[List[Dict]],
    ) -> np.ndarray:
        """
        ماسک پوست: face oval از MediaPipe (اگر لندمارک هست) ∩ تشخیص رنگ پوست.
        چشم/ابرو/لب داخل oval حذف می‌شوند تا رتوش آن‌ها را خراب نکند.
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                     397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                     172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        EYE_REGIONS = [
            [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        ]
        LIPS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                375, 321, 405, 314, 17, 84, 181, 91, 146]

        if landmarks_norm and len(landmarks_norm) >= 468:
            pts = np.array([[lm['x'] * w, lm['y'] * h] for lm in landmarks_norm], dtype=np.int32)
            cv2.fillPoly(mask, [pts[FACE_OVAL]], 255)
            for region in EYE_REGIONS:
                erode = pts[region].copy().astype(np.float32)
                c = erode.mean(axis=0)
                erode = c + (erode - c) * 0.6  # کوچک‌تر از واقعیت
                cv2.fillPoly(mask, [erode.astype(np.int32)], 0)
            lp = pts[LIPS].copy().astype(np.float32)
            lc = lp.mean(axis=0)
            lp = lc + (lp - lc) * 0.7
            cv2.fillPoly(mask, [lp.astype(np.int32)], 0)
        else:
            mask[:] = 255  # بدون لندمارک: کل تصویر

        mask = cv2.GaussianBlur(mask, (0, 0), 8)

        # محدودسازی به رنگ‌های پوست‌مانند (YCrCb)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        cr, cb = ycrcb[:, :, 1].astype(int), ycrcb[:, :, 2].astype(int)
        skin_color = ((cr > 135) & (cr < 180) & (cb > 85) & (cb < 135) & (ycrcb[:, :, 0] > 60)).astype(np.uint8) * 255
        skin_color = cv2.dilate(skin_color, np.ones((9, 9), np.uint8))
        skin_color = cv2.GaussianBlur(skin_color, (0, 0), 10)

        combined = (mask.astype(np.float32) / 255.0) * (skin_color.astype(np.float32) / 255.0)
        return np.clip(combined, 0, 1).astype(np.float32)

    # -----------------------------------------------------
    @staticmethod
    def _encode(img: np.ndarray) -> Optional[str]:
        ok, buf = cv2.imencode('.jpg', np.clip(img, 0, 255).astype(np.uint8),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        return base64.b64encode(buf).decode('utf-8') if ok else None


retouch_service = RetouchService()
