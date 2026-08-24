"""
آناتومی بینی — مدل ۲۰ نقطه‌ای کلینیکی (Nasal Anatomy Landmarks)
================================================================
بر اساس آناتومی جراحی رینوپلاستی (همان ساختاری که در متن ارسالی آمده):

  ۱  نوک بینی (Nose Tip)          → ارتفاع و برجستگی
  ۲  پل بینی (Radix/Bridge)       → ارتفاع پل
  ۳  قوس بینی (Dorsum)            → انحنای خط مرکزی
  ۴  نوک داخلی (Infratip)         → زاویه نوک
  ۵  کولوملا (Columella)          → زاویه پایه
  ۶  سوراخ چپ (Nostril L)         → عرض/تقارن
  ۷  سوراخ راست (Nostril R)
  ۸  بال چپ (Alar L)              → عرض بینی
  ۹  بال راست (Alar R)
  ۱۰ پایه چپ (Alar Base L)        → زاویه پایه
  ۱۱ پایه راست (Alar Base R)
  ۱۲ گوشه چپ (Alar Crease L)      → عمق شیار
  ۱۳ گوشه راست (Alar Crease R)
  ۱۴ وسط پل (Mid Bridge)          → انحنای قوس
  ۱۵ نوک قوس (Tip Defining Point)
  ۱۶ بالای قوس (Supratip)
  ۱۷ پایین قوس (Infratip Break)
  ۱۸ دیواره چپ (Sidewall L)       → تقارن
  ۱۹ دیواره راست (Sidewall R)
  ۲۰ مرکز پایه (Base Center)

نگاشت به MediaPipe FaceMesh (اندیس‌های رسمی 468 نقطه):
  هر نقطه آناتومیک = میانگین وزنی چند ورتکس FaceMesh
  (چون FaceMesh مستقیماً «سوراخ بینی» ندارد، از خوشه نقاط اطراف ساخته می‌شود)
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------
# خوشه‌های MediaPipe برای هر جزء آناتومیک
# (اندیس‌های رسمی FACEMESH — قابل اعتماد و پایدار)
# ---------------------------------------------------------
MP_CLUSTERS: Dict[str, List[int]] = {
    # ۱ نوک بینی: نقطه 4 + نقاط تعریف نوک
    'tip':            [4, 1, 5],
    # ۲ پل/رادیکس: ریشه بینی بین ابروها
    'radix':          [168, 8, 9],
    # ۳ قوس بینی (خط مرکزی dorsum)
    'dorsum':         [197, 195, 5],
    # ۴ اینفراتیپ (زیر نوک)
    'infratip':       [2, 94],
    # ۵ کولوملا (ستون بینی بین سوراخ‌ها)
    'columella':      [94, 2, 164],
    # ۶/۷ سوراخ‌ها: خوشه نقاط حلقوی سوراخ
    'nostril_l':      [129, 240, 49, 131, 98],
    'nostril_r':      [358, 460, 279, 360, 327],
    # ۸/۹ بال‌ها (alar rim)
    'alar_l':         [129, 98, 97, 206],
    'alar_r':         [358, 327, 326, 426],
    # ۱۰/۱۱ پایه بال‌ها (جایی که بال به صورت می‌چسبد)
    'alar_base_l':    [50, 205, 207],
    'alar_base_r':    [280, 425, 427],
    # ۱۲/۱۳ شیار بال‌ها (alar crease)
    'alar_crease_l':  [205, 50],
    'alar_crease_r':  [425, 280],
    # ۱۴ وسط پل
    'mid_bridge':     [197, 195],
    # ۱۵ نقطه تعریف نوک
    'tip_defining':   [4, 5],
    # ۱۶ سوپراتیپ (بالای نوک)
    'supratip':       [195, 5],
    # ۱۷ اینفراتیپ‌بریک (پایین نوک)
    'infratip_break': [2, 94, 19],
    # ۱۸/۱۹ دیواره‌ها (sidewall)
    'sidewall_l':     [48, 115, 220, 45],
    'sidewall_r':     [278, 344, 440, 275],
    # ۲۰ مرکز پایه
    'base_center':    [94, 2, 168],
}

# ترتیب استاندارد ۲۰ نقطه آناتومیک
ANATOMY_ORDER = [
    'tip', 'radix', 'dorsum', 'infratip', 'columella',
    'nostril_l', 'nostril_r', 'alar_l', 'alar_r',
    'alar_base_l', 'alar_base_r', 'alar_crease_l', 'alar_crease_r',
    'mid_bridge', 'tip_defining', 'supratip', 'infratip_break',
    'sidewall_l', 'sidewall_r', 'base_center',
]


class NoseAnatomy:
    """
    استخراج ۲۰ نقطه آناتومیک بینی از لندمارک‌های کامل FaceMesh.
    اگر FaceMesh نبود، از پلی‌گان ۱۸ نقطه‌ای قدیمی هم تخمین می‌زند.
    """

    def __init__(self, landmarks: Optional[List[Dict]] = None,
                 fallback_polygon: Optional[List[List[int]]] = None,
                 image_shape: Optional[Tuple[int, int]] = None):
        self.valid = False
        self.points: Dict[str, np.ndarray] = {}

        if landmarks and len(landmarks) >= 468 and image_shape is not None:
            h, w = image_shape[:2]
            first = landmarks[0]
            fx, fy = float(first['x']), float(first['y'])
            if 0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0 and w > 2 and h > 2:
                # مختصات نرمال (0..1) → پیکسل
                pts = np.array([[lm['x'] * w, lm['y'] * h] for lm in landmarks],
                               dtype=np.float32)
            else:
                # مختصات پیکسلی → مستقیم
                pts = np.array([[lm['x'], lm['y']] for lm in landmarks],
                               dtype=np.float32)
            for name, idxs in MP_CLUSTERS.items():
                self.points[name] = pts[idxs].mean(axis=0)
            self.valid = True
        elif fallback_polygon and len(fallback_polygon) == 18:
            self._from_polygon(np.array(fallback_polygon, dtype=np.float32))

    # -----------------------------------------------------
    def _from_polygon(self, p: np.ndarray):
        """تخمین آناتومی از پلی‌گان ۱۸ نقطه‌ای قدیمی."""
        tip = p[5]; radix = p[0]; bottom = p[9]
        span = bottom - radix
        self.points = {
            'tip': tip,
            'radix': radix,
            'dorsum': (radix + tip) / 2,
            'infratip': (tip + bottom) / 2,
            'columella': p[6],
            'nostril_l': (p[10] + p[12]) / 2,
            'nostril_r': (p[11] + p[13]) / 2,
            'alar_l': p[16],
            'alar_r': p[17],
            'alar_base_l': p[16] + 0.15 * span,
            'alar_base_r': p[17] + 0.15 * span,
            'alar_crease_l': p[16] - 0.05 * span,
            'alar_crease_r': p[17] - 0.05 * span,
            'mid_bridge': (radix + tip) / 2,
            'tip_defining': tip,
            'supratip': (tip + (radix + tip) / 2) / 2,
            'infratip_break': (tip + p[6]) / 2,
            'sidewall_l': (p[16] + tip) / 2,
            'sidewall_r': (p[17] + tip) / 2,
            'base_center': bottom,
        }
        self.valid = True

    # -----------------------------------------------------
    @property
    def alar_mid(self):
        """مرکز واقعی پایه بینی — میانه دو بال."""
        al, ar = self.points.get('alar_l'), self.points.get('alar_r')
        if al is None or ar is None:
            return self.points.get('base_center', np.zeros(2, np.float32))
        return (al + ar) / 2.0

    def get(self, name: str) -> Optional[np.ndarray]:
        return self.points.get(name)

    def ordered_array(self) -> Optional[np.ndarray]:
        if not self.valid:
            return None
        return np.stack([self.points[n] for n in ANATOMY_ORDER])

    # -----------------------------------------------------
    #   اندازه‌های کلینیکی
    # -----------------------------------------------------
    @property
    def nasal_width(self) -> float:
        """عرض بینی = فاصله دو بال (alar to alar) — Farkas canon"""
        return float(np.linalg.norm(self.points['alar_r'] - self.points['alar_l']))

    @property
    def nasal_height(self) -> float:
        """ارتفاع بینی = رادیکس تا زیر نوک"""
        return float(np.linalg.norm(self.points['infratip'] - self.points['radix']))

    @property
    def tip_projection_axis(self) -> np.ndarray:
        """محور برجستگی نوک (رادیکس → نوک)."""
        ax = self.points['tip'] - self.points['radix']
        return ax / max(np.linalg.norm(ax), 1e-6)

    @property
    def columella_axis(self) -> np.ndarray:
        """محور کولوملا (نوک → پایه) — زاویه nasolabial روی آن سنجیده می‌شود."""
        ax = self.points['base_center'] - self.points['tip']
        return ax / max(np.linalg.norm(ax), 1e-6)

    def width_ratio(self, intercanthal: Optional[float] = None) -> float:
        """نسبت عرض بینی به فاصله اینترکانتال (ایده‌آل ≈ 1.0)."""
        if intercanthal:
            return self.nasal_width / max(intercanthal, 1e-6)
        return 1.0

    def summary(self) -> Dict:
        if not self.valid:
            return {}
        return {
            'width': round(self.nasal_width, 1),
            'height': round(self.nasal_height, 1),
            'w_over_h': round(self.nasal_width / max(self.nasal_height, 1e-6), 3),
        }


def extract_anatomy(landmarks, image_shape) -> Optional[NoseAnatomy]:
    return NoseAnatomy(landmarks=landmarks, image_shape=image_shape)
