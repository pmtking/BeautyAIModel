"""
مدل‌های تخصصی Warping برای هر ناحیه صورت
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


# ============================================
# توابع کمکی
# ============================================

def _apply_radial_warp(image: np.ndarray, center: np.ndarray, max_dist: float,
                        scale: float) -> np.ndarray:
    """اعمال Warping شعاعی حول یک نقطه مرکزی"""
    h, w = image.shape[:2]
    if max_dist <= 0:
        return image

    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((y_idx - center[1]) ** 2 + (x_idx - center[0]) ** 2)

    inside = dist < max_dist
    factor = np.ones_like(dist, dtype=np.float32)
    factor[inside] = 1 - (dist[inside] / max_dist) * (1 - scale)

    map_x = center[0] + (x_idx - center[0]) * factor
    map_y = center[1] + (y_idx - center[1]) * factor

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def _apply_directional_warp(image: np.ndarray, anchor: np.ndarray, radius: float,
                             dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    """جابه‌جایی جهت‌دار حول یک نقطه"""
    h, w = image.shape[:2]
    if radius <= 0:
        return image

    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((y_idx - anchor[1]) ** 2 + (x_idx - anchor[0]) ** 2)

    weight = np.clip(1 - dist / radius, 0, 1) ** 2

    map_x = x_idx - dx * weight
    map_y = y_idx - dy * weight

    map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, h - 1).astype(np.float32)

    return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def _bbox_extent(points: np.ndarray) -> float:
    """اندازه‌ی تقریبی ناحیه"""
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    return float(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2))


# ============================================
# بینی
# ============================================

class NoseWarping:
    @staticmethod
    def smaller(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        scale = 1 - (intensity * 0.3)
        return _apply_radial_warp(image, center, max_dist, scale)

    @staticmethod
    def bigger(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        scale = 1 + (intensity * 0.3)
        return _apply_radial_warp(image, center, max_dist, scale)

    @staticmethod
    def slim_bridge(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        extent = _bbox_extent(pts)
        tip = pts[np.argmax(pts[:, 1])]

        result = _apply_radial_warp(image, center, extent * 0.5, 1 - intensity * 0.25)
        result = _apply_directional_warp(result, tip, radius=extent * 0.3,
                                          dx=0, dy=intensity * extent * 0.12)
        return result

    @staticmethod
    def doll_tip(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        extent = _bbox_extent(pts)
        tip = pts[np.argmax(pts[:, 1])]

        result = _apply_radial_warp(image, center, extent * 0.55, 1 - intensity * 0.35)
        result = _apply_radial_warp(result, tip, extent * 0.25, 1 - intensity * 0.3)
        result = _apply_directional_warp(result, tip, radius=extent * 0.3,
                                          dx=0, dy=intensity * extent * 0.15)
        return result

    @staticmethod
    def natural(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        extent = _bbox_extent(pts)
        return _apply_radial_warp(image, center, extent * 0.5, 1 - intensity * 0.1)


# ============================================
# لب (نسخه اصلاح‌شده با تفکیک بالا و پایین)
# ============================================

class LipWarping:
    @staticmethod
    def fuller(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        """حجم‌دهی کامل لب با تفکیک بالا و پایین"""
        pts = np.array(points, dtype=np.float32)
        
        if len(pts) < 3:
            return image
        
        # تفکیک لب بالا و پایین بر اساس Y
        y_min = pts[:, 1].min()
        y_max = pts[:, 1].max()
        y_mid = (y_min + y_max) / 2
        
        upper_pts = pts[pts[:, 1] < y_mid]
        lower_pts = pts[pts[:, 1] >= y_mid]
        
        # مرکز هر بخش
        upper_center = upper_pts.mean(axis=0) if len(upper_pts) > 0 else pts.mean(axis=0)
        lower_center = lower_pts.mean(axis=0) if len(lower_pts) > 0 else pts.mean(axis=0)
        
        # شعاع هر بخش
        upper_dist = np.max(np.linalg.norm(upper_pts - upper_center, axis=1)) if len(upper_pts) > 0 else 1
        lower_dist = np.max(np.linalg.norm(lower_pts - lower_center, axis=1)) if len(lower_pts) > 0 else 1
        
        scale = 1 + (intensity * 0.3)
        
        result = image.copy()
        
        # لب بالا
        if len(upper_pts) > 2:
            result = _apply_radial_warp(result, upper_center, upper_dist * 0.6, scale)
        
        # لب پایین
        if len(lower_pts) > 2:
            result = _apply_radial_warp(result, lower_center, lower_dist * 0.6, scale)
        
        return result

    @staticmethod
    def thinner(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        scale = 1 - (intensity * 0.25)
        return _apply_radial_warp(image, center, max_dist, scale)

    @staticmethod
    def heart_shape(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        extent = _bbox_extent(pts)
        cupid_point = pts[np.argmin(pts[:, 1])]
        lower_point = pts[np.argmax(pts[:, 1])]

        result = _apply_directional_warp(image, cupid_point, radius=extent * 0.3,
                                          dx=0, dy=intensity * extent * 0.06)
        result = _apply_radial_warp(result, lower_point, extent * 0.45, 1 + intensity * 0.35)
        return result

    @staticmethod
    def russian(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        extent = _bbox_extent(pts)

        result = _apply_directional_warp(image, center, radius=extent * 0.5,
                                          dx=0, dy=-intensity * extent * 0.1)
        return result

    @staticmethod
    def natural(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist, 1 + intensity * 0.12)


# ============================================
# فک
# ============================================

class JawWarping:
    @staticmethod
    def sharper(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        bottom = pts[-5:].mean(axis=0) if len(pts) >= 5 else pts.mean(axis=0)
        extent = _bbox_extent(pts)
        return _apply_directional_warp(image, bottom, radius=extent * 0.5,
                                        dx=0, dy=-intensity * extent * 0.1)

    @staticmethod
    def rounder(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        bottom = pts[-5:].mean(axis=0) if len(pts) >= 5 else pts.mean(axis=0)
        extent = _bbox_extent(pts)
        return _apply_directional_warp(image, bottom, radius=extent * 0.5,
                                        dx=0, dy=intensity * extent * 0.08)


# ============================================
# گونه
# ============================================

class CheekWarping:
    @staticmethod
    def enhance(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist, 1 + intensity * 0.2)

    @staticmethod
    def reduce(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist, 1 - intensity * 0.2)


# ============================================
# پیشانی
# ============================================

class ForeheadWarping:
    @staticmethod
    def smooth(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist, 1 - intensity * 0.1)

    @staticmethod
    def enhance(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist, 1 + intensity * 0.1)


# ============================================
# کلاس اصلی
# ============================================

class SpecializedWarping:
    def __init__(self):
        self.handlers = {
            'nose': {
                'smaller': NoseWarping.smaller,
                'bigger': NoseWarping.bigger,
                'slim_bridge': NoseWarping.slim_bridge,
                'doll_tip': NoseWarping.doll_tip,
                'natural': NoseWarping.natural,
            },
            'lip': {
                'fuller': LipWarping.fuller,
                'thinner': LipWarping.thinner,
                'heart_shape': LipWarping.heart_shape,
                'russian': LipWarping.russian,
                'natural': LipWarping.natural,
            },
            'jaw': {
                'sharper': JawWarping.sharper,
                'rounder': JawWarping.rounder
            },
            'cheek': {
                'enhance': CheekWarping.enhance,
                'reduce': CheekWarping.reduce
            },
            'forehead': {
                'smooth': ForeheadWarping.smooth,
                'enhance': ForeheadWarping.enhance
            }
        }
        logger.info("✅ SpecializedWarping initialized")

    def warp(self, image: np.ndarray, points: List[List[int]],
              area: str, action: str, intensity: float) -> np.ndarray:
        area_handlers = self.handlers.get(area, {})
        handler = area_handlers.get(action)

        if handler:
            return handler(image, points, intensity)
        else:
            logger.warning(f"No handler for area={area}, action={action}")
            return image

    def get_available_actions(self, area: str) -> List[str]:
        return list(self.handlers.get(area, {}).keys())


specialized_warping = SpecializedWarping()