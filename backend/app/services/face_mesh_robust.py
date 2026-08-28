# app/services/face_mesh_robust.py
"""
تشخیص لندمارک صورت با تحمل شرایط سخت:
  - نور کم / کنتراست پایین  → CLAHE + brighten
  - عکس کوچک/تار            → upscale هوشمند
  - چرخش EXIF اشتباه        → امتحان ۴ جهت
  - آستانه اطمینان پله‌ای   → 0.5 → 0.3 → 0.15

خروجی همیشه مختصات نرمال (0..1) نسبت به عکس اصلی است،
پس فراخوان‌ها هیچ تبدیلی لازم ندارند.
"""
import cv2
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple

# استفاده از سازگارکننده MediaPipe (Tasks API در نسخه‌های جدید)
from app.services.mp_shim import face_mesh as mp_face_mesh

logger = logging.getLogger(__name__)


class RobustFaceMesh:

    def __init__(self):
        self._mp = None

    def _make_fm(self, conf: float):
        return mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=conf,
        )

    # ---------------------------------------------------------
    def detect(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[Optional[List[Dict]], Dict]:
        """
        خروجی: (landmarks یا None، متادیتا)
        landmarks: [{'x':0..1,'y':0..1,'z':..}, ...] نسبت به عکس اصلی
        """
        h, w = image_bgr.shape[:2]
        meta: Dict = {'attempts': []}

        # ---------- آماده‌سازی کاندیدها ----------
        variants: List[Tuple[str, np.ndarray, str]] = []

        work = image_bgr
        if max(h, w) < 480:
            s = 640.0 / max(h, w)
            work = cv2.resize(image_bgr, None, fx=s, fy=s,
                              interpolation=cv2.INTER_CUBIC)
        variants.append(('plain', work, 'none'))

        # CLAHE روی کانال L → کنتراست محلی برای نور بد
        try:
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            variants.append(('clahe', cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 'none'))
        except Exception:
            pass

        # روشن‌سازی ساده
        bright = cv2.convertScaleAbs(work, alpha=1.3, beta=30)
        variants.append(('bright', bright, 'none'))

        # چرخش‌ها روی نسخه پایه (EXIF خراب)
        for code, tag in [(cv2.ROTATE_90_CLOCKWISE, 'rot90cw'),
                          (cv2.ROTATE_180, 'rot180'),
                          (cv2.ROTATE_90_COUNTERCLOCKWISE, 'rot90ccw')]:
            variants.append((tag, cv2.rotate(image_bgr, code), tag))

        # ---------- تلاش پله‌ای ----------
        for name, img_v, rot in variants:
            for conf in (0.5, 0.3, 0.15):
                try:
                    fm = self._make_fm(conf)
                    res = fm.process(cv2.cvtColor(img_v, cv2.COLOR_BGR2RGB))
                    fm.close()
                except Exception as e:
                    logger.warning(f'facemesh error ({name}/{conf}): {e}')
                    continue

                if not res.multi_face_landmarks:
                    meta['attempts'].append(f'{name}@{conf}:miss')
                    continue

                lms = [
                    {'x': lm.x, 'y': lm.y, 'z': lm.z}
                    for lm in res.multi_face_landmarks[0].landmark
                ]
                lms = self._unrotate(lms, rot)
                meta.update({
                    'variant': name,
                    'rotation': rot,
                    'confidence': conf,
                    'ok': True,
                })
                logger.info(f'face locked via {name}@{conf}')
                return lms, meta

        meta['ok'] = False
        return None, meta

    # ---------------------------------------------------------
    @staticmethod
    def _unrotate(lms: List[Dict], rot: str) -> List[Dict]:
        """تبدیل مختصات نرمال از قاب چرخیده به قاب اصلی."""
        if rot == 'none':
            return lms
        out = []
        for lm in lms:
            x, y = lm['x'], lm['y']
            if rot == 'rot90cw':       # inverse of CLOCKWISE
                x_o, y_o = y, 1 - x
            elif rot == 'rot180':
                x_o, y_o = 1 - x, 1 - y
            elif rot == 'rot90ccw':
                x_o, y_o = 1 - y, x
            else:
                x_o, y_o = x, y
            out.append({'x': x_o, 'y': y_o, 'z': lm['z']})
        return out


robust_face_mesh = RobustFaceMesh()
