# app/utils/blending.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import cv2
import numpy as np


class FaceBlending:
    """کلاس ترکیب تصاویر"""

    def __init__(self):
        pass

    def blend_with_mask(self, original: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        ترکیب دو تصویر با استفاده از ماسک
        
        Args:
            original: تصویر اصلی
            warped: تصویر تغییر یافته
            mask: ماسک
            
        Returns:
            np.ndarray: تصویر ترکیب شده
        """
        # نرمال‌سازی ماسک
        mask_normalized = mask / 255.0
        mask_normalized = np.expand_dims(mask_normalized, axis=2)

        # ترکیب
        result = original * (1 - mask_normalized) + warped * mask_normalized
        result = result.astype(np.uint8)

        return result

    def poisson_blending(self, original: np.ndarray, warped: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        ترکیب با استفاده از Poisson Blending
        
        Args:
            original: تصویر اصلی
            warped: تصویر تغییر یافته
            mask: ماسک
            
        Returns:
            np.ndarray: تصویر ترکیب شده
        """
        try:
            h, w = original.shape[:2]
            
            # پیدا کردن مرکز ماسک
            moments = cv2.moments(mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = w // 2, h // 2

            # Poisson Blending
            result = cv2.seamlessClone(
                warped,
                original,
                mask,
                (cx, cy),
                cv2.NORMAL_CLONE
            )
            
            return result

        except Exception as e:
            # اگر Poisson Blending کار نکرد، از روش ساده استفاده کن
            print(f"⚠️ Poisson blending failed, using simple blend: {e}")
            return self.blend_with_mask(original, warped, mask)