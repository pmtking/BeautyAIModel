# app/utils/warping.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import cv2
import numpy as np


class FaceWarping:
    """کلاس تغییر شکل صورت"""
    
    def __init__(self):
        pass

    def warp_nose(self, image: np.ndarray, nose_points: list, intensity: float = 0.5) -> np.ndarray:
        """
        تغییر فرم بینی
        
        Args:
            image: تصویر ورودی
            nose_points: نقاط کلیدی بینی
            intensity: شدت تغییر (0.0 تا 1.0)
            
        Returns:
            np.ndarray: تصویر تغییر یافته
        """
        h, w = image.shape[:2]
        
        # تبدیل نقاط به آرایه numpy
        pts = np.array(nose_points, dtype=np.float32)
        
        # محاسبه مرکز بینی
        center = np.mean(pts, axis=0).astype(int)
        
        # محاسبه حداکثر فاصله از مرکز
        max_dist = 0
        for pt in pts:
            dist = np.sqrt((pt[0] - center[0])**2 + (pt[1] - center[1])**2)
            max_dist = max(max_dist, dist)
        
        if max_dist == 0:
            return image
        
        # ایجاد نقشه تغییر
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # اعمال تغییر بر اساس فاصله از مرکز
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                
                if dist < max_dist:
                    # محاسبه ضریب تغییر
                    factor = 1 - (dist / max_dist) * intensity * 0.3
                    
                    # محاسبه موقعیت جدید
                    new_x = center[0] + (j - center[0]) * factor
                    new_y = center[1] + (i - center[1]) * factor
                    
                    # محدود کردن به محدوده تصویر
                    new_x = max(0, min(w - 1, new_x))
                    new_y = max(0, min(h - 1, new_y))
                    
                    map_x[i, j] = new_x
                    map_y[i, j] = new_y
        
        # اعمال نقشه
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped

    def warp_lip(self, image: np.ndarray, lip_points: list, intensity: float = 0.5) -> np.ndarray:
        """
        تغییر فرم لب
        
        Args:
            image: تصویر ورودی
            lip_points: نقاط کلیدی لب
            intensity: شدت تغییر (0.0 تا 1.0)
            
        Returns:
            np.ndarray: تصویر تغییر یافته
        """
        h, w = image.shape[:2]
        
        # تبدیل نقاط به آرایه numpy
        pts = np.array(lip_points, dtype=np.float32)
        
        # محاسبه مرکز لب
        center = np.mean(pts, axis=0).astype(int)
        
        # محاسبه حداکثر فاصله از مرکز
        max_dist = 0
        for pt in pts:
            dist = np.sqrt((pt[0] - center[0])**2 + (pt[1] - center[1])**2)
            max_dist = max(max_dist, dist)
        
        if max_dist == 0:
            return image
        
        # ایجاد نقشه تغییر
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        
        # اعمال تغییر (بزرگ‌تر کردن لب)
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center[1])**2 + (j - center[0])**2)
                
                if dist < max_dist:
                    # محاسبه ضریب تغییر (برعکس بینی)
                    factor = 1 + (1 - dist / max_dist) * intensity * 0.2
                    
                    # محاسبه موقعیت جدید
                    new_x = center[0] + (j - center[0]) * factor
                    new_y = center[1] + (i - center[1]) * factor
                    
                    # محدود کردن به محدوده تصویر
                    new_x = max(0, min(w - 1, new_x))
                    new_y = max(0, min(h - 1, new_y))
                    
                    map_x[i, j] = new_x
                    map_y[i, j] = new_y
        
        # اعمال نقشه
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
        
        return warped

    def create_mask(self, image: np.ndarray, points: list) -> np.ndarray:
        """
        ایجاد ماسک برای ناحیه مورد نظر
        
        Args:
            image: تصویر ورودی
            points: نقاط کلیدی
            
        Returns:
            np.ndarray: ماسک
        """
        h, w = image.shape[:2]
        
        # تبدیل نقاط به آرایه
        pts = np.array(points, dtype=np.int32)
        
        # ایجاد ماسک
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        
        # نرم‌سازی لبه‌ها
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return mask