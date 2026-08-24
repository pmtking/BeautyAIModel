import cv2
import numpy as np
from typing import List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

try:
    try:
        from warping.specialized import SpecializedWarping  # package import
    except ImportError:
        from specialized import SpecializedWarping          # flat import
except ImportError:
    SpecializedWarping = None
    logger.warning("⚠️ SpecializedWarping not available")


class WarpingModel:
    def __init__(self):
        self.specialized = SpecializedWarping() if SpecializedWarping else None
        self.blur_radius = 31
        self.dilate_iterations = 1
        self.area_colors = {
            'nose': (0, 0, 255),
            'lip': (255, 0, 0),
            'jaw': (0, 255, 0),
            'cheek': (255, 255, 0),
            'eye': (255, 0, 255),
            'forehead': (0, 255, 255)
        }
        logger.info("✅ WarpingModel initialized")

    def warp(self, image: np.ndarray, points: Union[List, np.ndarray],
             intensity: float = 0.5, action: str = 'smaller', 
             area: str = 'nose', landmarks=None, image_shape=None) -> np.ndarray:
        if isinstance(points, np.ndarray):
            points = points.tolist()
            
        if not points or len(points) < 3:
            return image.copy()

        if self.specialized:
            try:
                return self.specialized.warp(image, points, area, action, intensity,
                                             landmarks=landmarks,
                                             image_shape=image_shape)
            except Exception as e:
                logger.warning(f"Specialized warping failed: {e}")

        return self._fallback_warp(image, points, intensity, action, area)

    def _fallback_warp(self, image: np.ndarray, points: List[List[int]], 
                       intensity: float, action: str, area: str) -> np.ndarray:
        h, w = image.shape[:2]
        pts = np.array(points, dtype=np.float32)
        
        if len(pts) < 3:
            return image.copy()
        
        center = np.mean(pts, axis=0)
        max_dist = np.max(np.linalg.norm(pts - center, axis=1))
        
        if max_dist < 1:
            return image.copy()
        
        if action in ['smaller', 'thinner', 'reduce', 'slim_bridge']:
            scale = 1 - (intensity * 0.35)
        elif action in ['bigger', 'fuller', 'enhance', 'heart_shape']:
            scale = 1 + (intensity * 0.35)
        else:
            scale = 1 + (intensity * 0.2)
        
        y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
        dist = np.sqrt((y_idx - center[1]) ** 2 + (x_idx - center[0]) ** 2)
        
        factor = np.ones_like(dist, dtype=np.float32)
        inside = dist < max_dist
        ratio = dist[inside] / max_dist
        factor[inside] = 1 - (ratio ** 1.5) * (1 - scale)
        
        map_x = center[0] + (x_idx - center[0]) * factor
        map_y = center[1] + (y_idx - center[1]) * factor
        
        map_x = np.clip(map_x, 0, w - 1).astype(np.float32)
        map_y = np.clip(map_y, 0, h - 1).astype(np.float32)
        
        warped = cv2.remap(image, map_x, map_y, cv2.INTER_CUBIC)
        return warped

    def create_mask(self, image: np.ndarray, points: Union[List, np.ndarray]) -> np.ndarray:
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        if isinstance(points, np.ndarray):
            points = points.tolist()
            
        if not points or len(points) < 3:
            return mask
        
        pts = np.array(points, dtype=np.float32)
        
        # 🎯 ضد-کجی: مرتب‌سازی زاویه‌ای نقاط حول مرکز — پلی‌گان‌های
        # نامرتب (مثل بینی MediaPipe با ترتیب L/R درهم) self-intersection
        # می‌سازند → مرکز ماسک جابجا → blend نوک را کج می‌کشد
        c = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        pts = pts[np.argsort(angles)]
        
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        # 🎯 بلور متناسب با اندازه ناحیه — نه ثابت!
        # قبلاً blur=31 روی بینیِ ۱۳۰ پیکسلی، هسته ماسک را هم محو می‌کرد
        # (۸۹٪ افت اثر!) → حالا ~۶٪ اندازه ناحیه، حداقل ۵ حداکثر ۲۱
        area_px = float(cv2.countNonZero(mask))
        side = max(np.sqrt(area_px), 12.0)
        k = int(max(5, min(21, side * 0.16)))
        if k % 2 == 0:
            k += 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        return mask

    def draw_area(self, image: np.ndarray, points: Union[List, np.ndarray], 
                  area: str) -> np.ndarray:
        result = image.copy()
        
        if isinstance(points, np.ndarray):
            points = points.tolist()
            
        if not points or len(points) < 3:
            return result
        
        pts = np.array(points, dtype=np.int32)
        color = self.area_colors.get(area, (0, 255, 0))
        
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
        
        for p in pts:
            cv2.circle(result, tuple(p), 2, color, -1)
        
        return result