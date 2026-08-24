"""
مدل‌های تخصصی Warping برای هر ناحیه صورت
=========================================
بینی: از nose_styles.py (استایل‌های دقیق با لنگرهای MediaPipe)
لب:   از lip_styles.py  (استایل‌های روسی/برزیلی/هالیوودی/قلوه‌ای و ...)
سایر نواحی: radial/directional ساده
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import logging

from .nose_styles import (
    NoseStyles, NoseAnatomyStyles, NoseWarping, Maneuvers, _resolve
)
from .lip_styles import LipStyles, LipGeo, LipWarping
try:
    from .nose_anatomy import NoseAnatomy, MP_CLUSTERS, ANATOMY_ORDER
except Exception:
    NoseAnatomy = None

logger = logging.getLogger(__name__)


# ============================================
# توابع کمکی (برای سایر نواحی + سازگاری قدیمی)
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
# بینی → استایل‌های دقیق nose_styles
# ============================================
# (NoseWarping در nose_styles.py تعریف شده و از همان‌جا import می‌شود)


# ============================================
# لب → استایل‌های دقیق lip_styles
# ============================================
# (LipWarping در lip_styles.py تعریف شده و از همان‌جا import می‌شود)


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
    def smaller(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        """کوچک‌سازی چونه/فک: جمع‌کردن شعاعی ملایم حول مرکز ناحیه."""
        pts = np.array(points, dtype=np.float32)
        center = pts.mean(axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        return _apply_radial_warp(image, center, max_dist * 1.15,
                                  1 - intensity * 0.22)

    @staticmethod
    def rounder(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        bottom = pts[-5:].mean(axis=0) if len(pts) >= 5 else pts.mean(axis=0)
        extent = _bbox_extent(pts)
        return _apply_directional_warp(image, bottom, radius=extent * 0.5,
                                       dx=0, dy=intensity * extent * 0.08)

    @staticmethod
    def wider(image: np.ndarray, points: List[List[int]], intensity: float) -> np.ndarray:
        pts = np.array(points, dtype=np.float32)
        bottom = pts[-5:].mean(axis=0) if len(pts) >= 5 else pts.mean(axis=0)
        extent = _bbox_extent(pts)
        return _apply_directional_warp(image, bottom, radius=extent * 0.55,
                                       dx=intensity * extent * 0.06,
                                       dy=-intensity * extent * 0.04)


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
        # 🆕 بینی: موتور آناتومیک — امضا (image, landmarks, shape, intensity)
        self.anatomy_handlers = {
            'smaller': NoseAnatomyStyles.smaller,
            'bigger': NoseAnatomyStyles.bigger,
            'narrower': NoseAnatomyStyles.narrower,
            'wider': NoseAnatomyStyles.wider,
            'longer': NoseAnatomyStyles.longer,
            'shorter': NoseAnatomyStyles.shorter,
            'upturned_tip': NoseAnatomyStyles.upturned_tip,
            'droopy_tip': NoseAnatomyStyles.droopy_tip,
            'doll_tip': NoseAnatomyStyles.doll_tip,
            'fleshy': NoseAnatomyStyles.fleshy,
            'bony': NoseAnatomyStyles.bony,
            'fantasy': NoseAnatomyStyles.fantasy,
            'half_fantasy': NoseAnatomyStyles.half_fantasy,
            'natural': NoseAnatomyStyles.natural,
            'ideal_realistic': NoseAnatomyStyles.ideal_realistic,
            'filler': NoseAnatomyStyles.filler,
            'slim_bridge': NoseAnatomyStyles.slim_bridge,
            'hump_reduction': NoseAnatomyStyles.hump_reduction,
        }
        # legacy handlers (points-based) برای سایر نواحی
        self.handlers = {
            'nose': {},   # بینی از anatomy_handlers عبور می‌کند
            'lip': {
                'fuller': LipStyles.fuller,
                'thinner': LipStyles.thinner,
                'natural': LipStyles.natural,
                'russian': LipStyles.russian,
                'brazilian': LipStyles.brazilian,
                'hollywood': LipStyles.hollywood,
                'heart_shape': LipStyles.heart_shape,
                'classic': LipStyles.classic,
                'cupids_bow': LipStyles.cupids_bow,
                'corner_lift': LipStyles.corner_lift,
            },
            'jaw': {
                'sharper': JawWarping.sharper,
                'rounder': JawWarping.rounder,
                'wider': JawWarping.wider,
            },
            'cheek': {
                'enhance': CheekWarping.enhance,
                'reduce': CheekWarping.reduce,
            },
            'forehead': {
                'smooth': ForeheadWarping.smooth,
                'enhance': ForeheadWarping.enhance,
            }
        }
        logger.info("✅ SpecializedWarping initialized (anatomy engine)")

    def warp(self, image: np.ndarray, points: List[List[int]],
             area: str, action: str, intensity: float,
             landmarks=None, image_shape=None) -> np.ndarray:
        """landmarks + image_shape → فعال‌سازی موتور آناتومیک بینی."""
        if area == 'nose' and landmarks is not None and image_shape is not None:
            handler = self.anatomy_handlers.get(action)
            if handler:
                try:
                    return handler(image, landmarks, image_shape, intensity)
                except Exception as e:
                    logger.warning(f"anatomy nose warp failed ({action}): {e}")
                    return image
            logger.warning(f"No anatomy handler for nose action={action}")
            return image

        area_handlers = self.handlers.get(area, {})
        handler = area_handlers.get(action)

        if handler:
            return handler(image, points, intensity)
        else:
            logger.warning(f"No handler for area={area}, action={action}")
            return image

    def get_available_actions(self, area: str) -> List[str]:
        if area == 'nose':
            return list(self.anatomy_handlers.keys())
        return list(self.handlers.get(area, {}).keys())


specialized_warping = SpecializedWarping()
