import cv2
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BlendingModel:
    def __init__(self):
        self.blur_radius = 51
        logger.info("✅ BlendingModel initialized")

    def blend(self, original: np.ndarray, warped: np.ndarray, 
              mask: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        ترکیب تصویر اصلی و تغییر یافته برای تمام نقاط صورت
        
        Args:
            original: تصویر اصلی (BGR)
            warped: تصویر تغییر یافته (BGR)
            mask: ماسک ناحیه (uint8)
            method: روش ترکیب ('poisson' یا 'alpha')
        
        Returns:
            تصویر ترکیب شده
        """
        h, w = original.shape[:2]
        
        # اطمینان از یکسان بودن ابعاد
        if warped.shape != original.shape:
            warped = cv2.resize(warped, (w, h), interpolation=cv2.INTER_CUBIC)
        
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        
        # روش Poisson برای کیفیت بهتر
        if method == 'poisson':
            try:
                return self._poisson_blend(original, warped, mask)
            except Exception as e:
                logger.warning(f"Poisson failed: {e}, using alpha")
                return self._alpha_blend(original, warped, mask)
        
        # روش Alpha برای همه حالات
        return self._alpha_blend(original, warped, mask)

    def _poisson_blend(self, original: np.ndarray, warped: np.ndarray, 
                       mask: np.ndarray) -> np.ndarray:
        """Poisson Blending - بهترین کیفیت برای تمام نقاط"""
        h, w = original.shape[:2]
        
        # نرم‌سازی ماسک
        mask_blur = cv2.GaussianBlur(mask, (11, 11), 0)
        mask_blur = np.clip(mask_blur, 0, 255).astype(np.uint8)
        
        # پیدا کردن مرکز ماسک
        moments = cv2.moments(mask_blur)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            center = (cx, cy)
        else:
            center = (w // 2, h // 2)
        
        # تبدیل به RGB
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        
        # MIXED_CLONE برای ترکیب طبیعی‌تر
        result = cv2.seamlessClone(
            warped_rgb, original_rgb, mask_blur,
            center, cv2.MIXED_CLONE
        )
        
        return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    def _alpha_blend(self, original: np.ndarray, warped: np.ndarray, 
                     mask: np.ndarray) -> np.ndarray:
        """Alpha Blending - پایدار برای همه نقاط"""
        # نرم‌سازی ماسک با شعاع مناسب
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (self.blur_radius, self.blur_radius), 0)
        mask_3d = np.stack([mask_float] * 3, axis=2)
        
        # ترکیب
        original_float = original.astype(np.float32)
        warped_float = warped.astype(np.float32)
        
        result = original_float * (1 - mask_3d) + warped_float * mask_3d
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

    def create_smooth_mask(self, mask: np.ndarray, radius: Optional[int] = None) -> np.ndarray:
        """ایجاد ماسک نرم برای هر ناحیه"""
        if radius is None:
            radius = self.blur_radius
        
        mask_float = mask.astype(np.float32) / 255.0
        mask_float = cv2.GaussianBlur(mask_float, (radius, radius), 0)
        return (mask_float * 255).astype(np.uint8)

    def blend_with_edges(self, original: np.ndarray, warped: np.ndarray, 
                         mask: np.ndarray, edge_blur: int = 31) -> np.ndarray:
        """ترکیب با کنترل لبه‌ها - برای نواحی حساس"""
        # ماسک با لبه‌های نرم
        mask_smooth = self.create_smooth_mask(mask, edge_blur)
        
        # تبدیل به float
        mask_float = mask_smooth.astype(np.float32) / 255.0
        mask_3d = np.stack([mask_float] * 3, axis=2)
        
        # ترکیب
        original_float = original.astype(np.float32)
        warped_float = warped.astype(np.float32)
        
        result = original_float * (1 - mask_3d) + warped_float * mask_3d
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result