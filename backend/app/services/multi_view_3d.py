"""
بازسازی سه‌بعدی صورت از چند عکس (Multi-View → 3D Avatar)
========================================================
ورودی: ۱ تا ۴ عکس (جلو + نیم‌رخ چپ/راست)
فرایند:
  ۱. FaceMesh دقیق (refine_landmarks → 478 نقطه) روی هر عکس
  ۲. تخمین زاویه yaw هر نمای غیر روبرو (نسبت گوشه‌های چشم)
  ۳. هم‌ترازی Similarity (Procrustes) نمای جانبی → قاب نمای جلو
  ۴. فیوژن عمق: z هر ورتکس = میانگین وزنی z همه نماها
     + تقویت پروفایل: برجستگی سیلوئت نیم‌رخ (پل/نوک/چانه)
       به z نقاط خط وسط اضافه می‌شود ← جزئیات واقعی پروفایل
  ۵. مش: مثلث‌بندی Delaunay روی xy (OpenCV Subdiv2D)
  ۶. بافت: خود عکس جلو (UV = مختصات نرمال xy)
  ۷. رندر پیش‌نمایش چرخیده با سایه‌زنی Flat
"""
import cv2
import numpy as np
import base64
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# اندیس‌های پایدار برای هم‌ترازی (گوشه چشم‌ها + گودی گیجگاهی)
ALIGN_IDX = [33, 263, 127, 356]
MIDLINE_HINT = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 152, 200, 199]


class MultiViewReconstructor:

    def reconstruct(self, images: List[np.ndarray]) -> Dict:
        """images: لیست BGR — اولین عکس = نمای جلو (مرجع)."""
        try:
            from app.services.face_mesh_robust import robust_face_mesh

            if not images:
                return {'ok': False, 'error': 'هیچ تصویری ارسال نشده'}

            views = []
            for i, img in enumerate(images[:4]):
                lms, meta = robust_face_mesh.detect(img)
                if not lms:
                    continue
                h, w = img.shape[:2]
                pts = np.array([[lm['x'] * w, lm['y'] * h] for lm in lms], dtype=np.float32)
                zs = np.array([lm.get('z', 0.0) for lm in lms], dtype=np.float32) * w
                yaw = self._estimate_yaw(pts)
                views.append({'pts': pts, 'z': zs, 'yaw': yaw, 'shape': (h, w),
                              'variant': meta.get('variant'), 'conf': meta.get('confidence')})

            if not views:
                return {'ok': False, 'error': 'چهره در هیچ تصویری شناسایی نشد'}

            front = views[0]
            n = len(front['pts'])
            fused_z = front['z'].copy()
            weights = np.ones(n)

            # ---- فیوژن نماهای جانبی ----
            for v in views[1:]:
                T, inliers = self._align(v['pts'], front['pts'])
                if T is None:
                    continue
                aligned = cv2.transform(v['pts'][None, :, :], T)[0]
                # z نماهای مقابل باید علامتش برگردد (MediaPipe z نسبت به دوربین)
                z_side = v['z'] * (-1 if v['yaw'] * front['yaw'] < 0 else 1)
                conf = 0.45 + 0.25 * abs(np.cos(np.radians(v['yaw'])))
                fused_z += conf * z_side
                weights += conf

                # تقویت پروفایل از سیلوئت نیم‌رخ — وزن‌دار و محدود، نه خام
                boost = self._profile_boost(v, front)
                if boost is not None:
                    fused_z += boost * 0.35   # 🎯 ضریب کم: جزئیات، نه لایه دوم

            fused_z /= weights
            # 🎯 ضد-دوبله: هموارسازی دوطرفه قوی‌تر روی خط وسط + clip پرش‌ها
            fused_z = self._smooth_z(front['pts'], fused_z, k=9)
            fused_z = self._despeckle_z(front['pts'], fused_z)

            # 🎯 تقویت عمق: MediaPipe z فشرده است (برجستگی بینی را کم نشان می‌دهد)
            # نرمال‌سازی: z را به نسبت استاندارد آناتومیک باز-مقیاس می‌کنیم
            # تا بینی/گونه/چانه واقعاً برجسته دیده شوند
            fused_z = self._enhance_depth(front['pts'], fused_z)

            pts2d = front['pts']
            h, w = front['shape']

            # ---- مثلث‌بندی ----
            faces = self._triangulate(pts2d, w, h)
            if not len(faces):
                return {'ok': False, 'error': 'مثلث‌بندی ناموفق'}

            uvs = np.stack([pts2d[:, 0] / w, pts2d[:, 1] / h], axis=1)

            # ---- بافت و پیش‌نمایش ----
            ok, tex_buf = cv2.imencode('.jpg', images[0], [cv2.IMWRITE_JPEG_QUALITY, 92])
            texture_b64 = base64.b64encode(tex_buf).decode() if ok else None
            preview_b64 = self._render_preview(pts2d, fused_z, faces, size=720)

            return {
                'ok': True,
                'views_used': len(views),
                'yaws': [round(v['yaw'], 1) for v in views],
                'mesh': {
                    'vertices': np.round(
                        np.stack([pts2d[:, 0], pts2d[:, 1], fused_z], axis=1), 2
                    ).tolist(),
                    'uvs': np.round(uvs, 5).tolist(),
                    'faces': faces.tolist(),
                    'num_vertices': int(n),
                    'num_faces': int(len(faces)),
                },
                'texture': texture_b64,
                'preview': preview_b64,
            }
        except Exception as e:
            logger.exception('reconstruct failed')
            return {'ok': False, 'error': str(e)}

    # -----------------------------------------------------
    def _estimate_yaw(self, pts: np.ndarray) -> float:
        """زاویه تقریبی yaw از نسبت فاصله گوشه چشم تا کنار صورت."""
        L, R = pts[33], pts[263]
        earL, earR = pts[127], pts[356]
        dL = float(np.linalg.norm(L - earL))
        dR = float(np.linalg.norm(R - earR))
        ratio = (dR - dL) / max(dR + dL, 1e-6)          # -1..1
        return float(np.degrees(np.arcsin(np.clip(ratio, -0.99, 0.99))))

    def _align(self, src: np.ndarray, dst: np.ndarray) -> Tuple[Optional[np.ndarray], int]:
        s = src[ALIGN_IDX].astype(np.float32)
        d = dst[ALIGN_IDX].astype(np.float32)
        T, inliers = cv2.estimateAffinePartial2D(s, d, method=cv2.LMEDS)
        return T, 0 if inliers is None else int(inliers.sum())

    def _profile_boost(self, side: Dict, front: Dict) -> np.ndarray:
        """برجستگی سیلوئت نیم‌رخ (اختلاف x لبه نسبت به تناسب) → z نقاط میانی."""
        pts = side['pts'].copy()
        h, w = side['shape']
        boost = np.zeros(len(pts))

        ys = pts[:, 1]
        x_min_col = np.full(int(ys.max()) + 1, np.nan)
        for y in range(0, int(ys.max()), 4):
            sel = (ys >= y) & (ys < y + 4)
            if sel.any():
                x_min_col[y // 4 * 4:y // 4 * 4 + 4] = pts[sel][:, 0].min()

        valid = ~np.isnan(x_min_col)
        if valid.sum() < 20:
            return boost

        # انحنای پروفایل = انحراف لبه از خط صاف بین forehead و chin
        idxs = np.where(valid)[0]
        top, bot = idxs[0], idxs[-1]
        line = np.interp(idxs, [top, bot], [x_min_col[top], x_min_col[bot]])
        curve = line - x_min_col[idxs]                     # مثبت = بیرون‌زده
        curve = curve - curve.mean()                       # نرمال‌سازی

        # 🎯 ضد-دوبله: boost فقط مؤلفهٔ کم‌بسامد پروفایل (فرم بینی/چانه) —
        # نویز فرکانس بالا که «حاله دوم» می‌سازد حذف می‌شود
        try:
            k = max(5, (len(curve) // 8) | 1)
            kernel = np.ones(k) / k
            curve = np.convolve(curve, kernel, mode='same')
        except Exception:
            pass

        cx = front['pts'][:, 0].mean()
        mid = np.abs(front['pts'][:, 0] - cx) < 0.08 * w   # نقاط خط وسط صورت
        vy = front['pts'][:, 1].clip(top, bot)
        prof = np.interp(vy, idxs.astype(float), curve)

        # 🎯 سقف دامنه: boost هرگز نباید از ۳۵٪ انحراف z خودِ جلو بزرگتر شود
        cap = 0.35 * float(np.std(front['z'])) if len(front['z']) else 20.0
        prof = np.clip(prof, -cap, cap)
        boost[mid] = prof[mid]                              # ضریب بیرونی در caller
        return boost

    def _smooth_z(self, pts: np.ndarray, z: np.ndarray, k: int = 6) -> np.ndarray:
        """هموارسازی محلی z بین همسایه‌های نزدیک xy."""
        order = np.argsort(pts[:, 1])
        zs = z.copy().astype(np.float32)
        kernel = np.ones(k) / k
        zs_sorted = np.convolve(zs[order], kernel, mode='same')
        zs[order] = zs_sorted
        return zs

    def _despeckle_z(self, pts: np.ndarray, z: np.ndarray) -> np.ndarray:
        """🎯 ضد-دوبله: پرش‌های موضعی z (گوست) را با میانه همسایه‌ها جایگزین کن."""
        try:
            zs = z.copy().astype(np.float32)
            # شبکه‌بندی xy → میانه z در هر خانه
            x0, y0 = pts.min(axis=0)
            x1, y1 = pts.max(axis=0)
            gx = np.clip(((pts[:, 0] - x0) / max(x1 - x0, 1) * 40).astype(int), 0, 39)
            gy = np.clip(((pts[:, 1] - y0) / max(y1 - y0, 1) * 60).astype(int), 0, 59)
            grid_med = np.full((60, 40), np.nan)
            for ix, iy, zv in zip(gx, gy, zs):
                col = grid_med[iy, ix]
                grid_med[iy, ix] = zv if np.isnan(col) else (col + zv) / 2
            # پر کردن خانه‌های خالی با میانه همسایه
            valid = ~np.isnan(grid_med)
            if valid.sum() < 10:
                return zs
            med_fill = cv2.inpaint(
                (np.nan_to_num(grid_med, nan=0) * 255).astype(np.uint8),
                (~valid).astype(np.uint8), 3, cv2.INPAINT_NEIGHBORS
            ).astype(np.float32) / 255.0
            for i in range(len(zs)):
                zm = med_fill[gy[i], gx[i]]
                # اگر z از میانه خانه بیش از آستانه دور است → گوست، جایگزین کن
                if abs(zs[i] - zm) > 0.45 * max(np.std(zs), 1e-6):
                    zs[i] = zm
            return zs
        except Exception:
            return z

    def _triangulate(self, pts: np.ndarray, w: int, h: int) -> np.ndarray:
        """مثلث‌بندی Delaunay + تبدیل مختصات به اندیس ورتکس."""
        # نقاط یکتا را به اندیس نگاشت کن (Subdiv مختصات برمی‌گرداند!)
        coord_to_idx: Dict[Tuple[int, int], int] = {}
        for i, p in enumerate(pts):
            key = (int(round(p[0])), int(round(p[1])))
            coord_to_idx.setdefault(key, i)

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for p in pts:
            subdiv.insert((float(p[0]), float(p[1])))
        tri = subdiv.getTriangleList().astype(np.float32)

        keep = []
        n = len(pts)
        for t in tri:
            idxs = []
            ok = True
            for j in range(3):
                key = (int(round(t[2 * j])), int(round(t[2 * j + 1])))
                vi = coord_to_idx.get(key)
                if vi is None:
                    ok = False
                    break
                idxs.append(vi)
            if not ok:
                continue
            a, b, c = idxs
            pa, pb, pc = pts[a], pts[b], pts[c]
            area = 0.5 * abs(float(np.cross(pb - pa, pc - pa)))
            if area > 4:                                    # حذف مثلث‌های خرابی
                keep.append([a, b, c])
        return np.array(keep, dtype=np.int32)

    def _render_preview(self, pts: np.ndarray, z: np.ndarray,
                        faces: np.ndarray, size: int = 720) -> Optional[str]:
        """رندر سریع چرخیده ۲۵ درجه با سایه flat — پیش‌نمایش سه‌بعدی."""
        try:
            if not len(faces):
                return None
            canvas = np.zeros((size, size, 3), dtype=np.uint8)
            p = pts.copy()
            c = p.mean(axis=0)
            scale = (size * 0.82) / max(float(np.ptp(p[:, 0])),
                                        float(np.ptp(p[:, 1])), 1.0)
            p = (p - c) * scale
            zz = (z - z.mean()) * scale

            theta = np.radians(25)
            xr = p[:, 0] * np.cos(theta) + zz * np.sin(theta)
            zr = -p[:, 0] * np.sin(theta) + zz * np.cos(theta)
            proj = np.stack([xr + size / 2, p[:, 1] + size / 2], axis=1)

            light = np.array([0.3, -0.5, 0.8]); light /= np.linalg.norm(light)
            t = faces
            tz = zr[t].mean(axis=1)
            order = np.argsort(tz)[::-1]                    # دور → نزدیک

            # نرمال سه‌بعدی هر مثلث (xy از تصویر، z از عمق فیوژن‌شده)
            p3 = np.stack([pts[:, 0], pts[:, 1], zz], axis=1)
            v1 = p3[t[:, 1]] - p3[t[:, 0]]
            v2 = p3[t[:, 2]] - p3[t[:, 0]]
            normals = np.cross(v1, v2)
            nn = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
            shade = np.abs(nn @ light)
            base = np.asarray((150, 158, 170), dtype=np.float32)

            for i in order:
                col = tuple(int(min(255, ch)) for ch in (base * (0.35 + 0.65 * shade[i])))
                tri = proj[t[i]].astype(np.int32)
                cv2.fillPoly(canvas, [tri], col)
            return base64.b64encode(cv2.imencode('.jpg', canvas)[1]).decode()
        except Exception:
            logger.exception('render_preview failed')
            return None


multi_view_reconstructor = MultiViewReconstructor()
