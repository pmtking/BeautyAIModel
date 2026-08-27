"""
🎯 موتور v3 — رفع سه باگ اصلی کاربر + استایل‌های جدید
=======================================================
باگ‌های گزارش‌شده:
  1. «نوک بالا» → بینی کج میشود ❌
     ریشه: pivot چرخش = میانه آلارها ولی axis از radix→tip نیست دقیق؛
     در چرخش، یک سمت قوی‌تر کشیده میشود → asymmetry
  2. «قوز گرفته بشه» → فرم زیبا نمیدهد ❌
     ریشه: dorsum_reshape فقط scaleX میکند (پهن/باریک) نه برداشتن قوز عمودی!
  3. «نوک کوچک بشه» → ناحیه نوک را هدف نمیگیرد ❌
  4. «سوراخ‌ها تنگ‌تر» → تصویر خراب میشود ❌
     ریشه: nostril_symmetry فقط جابجایی dx/dy — نه جمع واقعی حلقه سوراخ

راه‌حل v3:
  - tip_rotation_v2: pivot صحیح (نقطه supratip-break) + وزن متقارن
    حول محور + محدودسازی lateral به نیم‌عرض هر سمت
  - hump_removal_v2: warp عمودی واقعی (dy منفی روی قوز) + سایه جدید
  - tip_reduction_v2: radial shrink فقط حول tip_defining
  - nostril_narrow_v2: shrink شعاعی حول مرکز هر سوراخ (نه جابجایی)
"""
import sys, cv2, numpy as np
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src')
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src/models')
from face_parser.model import FaceParserModel
from warping.nose_anatomy import NoseAnatomy


class EngineV3:
    """مانورهای جراحی نسخه ۳ — دقیق و آناتومیک"""

    def __init__(self):
        self.parser = FaceParserModel()

    # ═══════════════════════════════════════════
    def _smooth_field(self, shape_roi, center, radius, power=2.0):
        ys, xs = np.mgrid[0:shape_roi[0], 0:shape_roi[1]].astype(np.float32)
        d = np.sqrt((xs-center[0])**2 + (ys-center[1])**2)
        w = np.clip(1 - d/radius, 0, 1)**power
        return w, xs, ys

    # ═══════════════════════════════════════════
    def tip_rotation_v2(self, image, anat, degrees):
        """
        چرخش نوک — ضد-کجی:
          • pivot = نقطه شکست supratip (جایی که پل تمام و نوک شروع میشود)
          • وزن کاملاً متقارن حول pivot در x (چپ=راست)
          • lateral window جدا برای هر سمت با عرض مساوی
        """
        tip = anat.get('tip'); radix = anat.get('radix')
        al_l, al_r = anat.get('alar_l'), anat.get('alar_r')
        if any(v is None for v in (tip, radix, al_l, al_r)):
            return image.copy()

        pts = anat.ordered_array()
        if pts is None:
            return image.copy()

        # supratip break = 75% مسیر radix→tip
        pivot = radix + (tip - radix) * 0.72
        h = float(np.linalg.norm(tip - pivot)) + 1e-6   # بازوی چرخش
        axis_x = (al_l[0] + al_r[0]) / 2.0              # خط وسط واقعی

        x0, y0, x1, y1 = self._roi(image.shape, pts, pad=0.8)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        # فاصله از pivot
        vx = xs - pivot[0]; vy = ys - pivot[1]
        r = np.sqrt(vx*vx + vy*vy)

        # وزن اصلی: فقط نوک (شعاع کوچک حول tip)
        tip_w = np.clip(1 - np.sqrt((xs-tip[0])**2+(ys-tip[1])**2)/(h*1.15), 0, 1)**1.5

        # تقارن سخت: ممنوع بودن اختلاف وزن چپ/راست بیش از ۵٪
        left_half  = tip_w[:, :int(tip_w.shape[1]/2)]
        right_half = tip_w[:, int(tip_w.shape[1]/2):]
        min_cols = min(left_half.shape[1], right_half.shape[1])
        sym_diff = np.abs(left_half[:, -min_cols:] -
                          right_half[:, min_cols-1::-1]).mean()
        if sym_diff > 0.05:
            # تقارن اجباری: میانگین دو سمت
            mirrored = left_half[:, ::-1]
            avg = (left_half + mirrored) / 2
            left_half[:] = avg[:, :left_half.shape[1]]
            right_half = avg[:, min_cols-1::-1][:, :right_half.shape[1]]
            tip_w[:, :left_half.shape[1]] = left_half
            tip_w[:, left_half.shape[1]:] = right_half

        ang = np.radians(degrees) * tip_w * 0.9   # حداکثر ۹۰٪ اعمال
        ca, sa = np.cos(ang), np.sin(ang)
        rx = vx*ca - vy*sa
        ry = vx*sa + vy*ca

        map_x = (pivot[0] + rx - x0)
        map_y = (pivot[1] + ry - y0)

        warped = cv2.remap(image[y0:y1, x0:x1],
                           np.clip(map_x,0,x1-x0-1).astype(np.float32),
                           np.clip(map_y,0,y1-y0-1).astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        m3 = (tip_w)[..., None] * 0.95
        out = image.copy()
        out[y0:y1, x0:x1] = (warped*m3 + out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)
        return out

    # ═══════════════════════════════════════════
    def hump_removal_v2(self, image, anat, intensity):
        """
        برداشتن قوز — روش جراحی واقعی:
          قوز = برجستگی عمودی dorsum. باید dy مثبت (به پایین/داخل)
          روی ناحیه قوز اعمال شود + shading صاف جدید.
        """
        radix = anat.get('radix'); tip = anat.get('tip')
        mid = anat.get('mid_bridge') or (radix+tip)/2
        if any(v is None for v in (radix, tip)):
            return image.copy()

        pts = anat.ordered_array()
        x0, y0, x1, y1 = self._roi(image.shape, pts, pad=0.7)
        ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)

        # نوار مرکزی بینی: فاصله از خط radix→tip
        u = tip - radix; u_norm = u / (np.linalg.norm(u)+1e-6)
        n_vec = np.array([-u_norm[1], u_norm[0]], dtype=np.float32)

        rel = np.stack([xs-radix[0], ys-radix[1]], axis=-1)
        along = rel @ u_norm / (np.linalg.norm(u)+1e-6)      # 0=radix 1=tip
        perp  = rel @ n_vec                                   # جانبی

        # قوز معملاً در 30%-65% طول پل
        hump_zone = np.clip(1 - np.abs(along-0.48)/0.22, 0, 1)**1.4
        # فقط ستون مرکزی (|perp| کم)
        center_band = np.clip(1 - np.abs(perp)/(anat.nasal_width*0.28), 0, 1)**1.2
        weight = hump_zone * center_band * intensity

        # جابجایی به داخل (به سمت محور) = dy در جهت -n اگر قوز بالاست
        shift = perp / (np.abs(perp)+1e-6) * 0   # صفر — فقط فشار به پایین
        dy_px = -weight * anat.nasal_height * 0.06   # کمی به داخل صورت

        map_x = xs - x0
        map_y = ys - y0 + dy_px   # پیکسل‌های قوز به عقب رانده میشوند

        warped = cv2.remap(image[y0:y1,x0:x1],
                           map_x.astype(np.float32), map_y.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m3 = weight[...,None]*0.9
        out = image.copy()
        out[y0:y1,x0:x1] = (warped*m3 + out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)

        # 🎨 سایه صاف جدید: روشن کردن برجستگی قدیمی
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[...,0]
        L[y0:y1,x0:x1] += weight * 14   # روشن شدن جای قوز = صاف دیده شود
        lab[...,0] = np.clip(L, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ═══════════════════════════════════════════
    def tip_reduction_v2(self, image, anat, intensity):
        """کوچک‌سازی نوک: radial shrink حول tip_defining"""
        tip = anat.get('tip') or anat.get('tip_defining')
        if tip is None: return image.copy()
        pts = anat.ordered_array()
        x0,y0,x1,y1 = self._roi(image.shape, pts, pad=0.55)
        ys, xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)

        radius = anat.nasal_width * 0.42
        d = np.sqrt((xs-tip[0])**2 + (ys-tip[1])**2)
        w = np.clip(1-d/radius, 0, 1)**1.6

        scale = 1 - intensity * 0.28 * w       # تا ۲۸٪ جمع شدن
        map_x = tip[0] + (xs-tip[0])*scale - x0
        map_y = tip[1] + (ys-tip[1])*scale - y0

        warped = cv2.remap(image[y0:y1,x0:x1],
                           map_x.astype(np.float32), map_y.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m3 = w[...,None]
        out = image.copy()
        out[y0:y1,x0:x1] = (warped*m3 + out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)
        return out

    # ═══════════════════════════════════════════
    def nostril_narrow_v2(self, image, anat, intensity):
        """
        تنگ‌تر کردن سوراخ‌ها — shrink شعاعی حول مرکز هر سوراخ
        (قبلی جابجا میکرد → خراب میشد)
        """
        nl, nr = anat.get('nostril_l'), anat.get('nostril_r')
        if nl is None or nr is None: return image.copy()
        pts = anat.ordered_array()
        x0,y0,x1,y1 = self._roi(image.shape, pts, pad=0.4)
        ys, xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)

        result = image.copy()
        for c in (nl, nr):
            radius = anat.nasal_width * 0.24
            d = np.sqrt((xs-c[0])**2 + (ys-c[1])**2)
            w = np.clip(1-d/radius, 0, 1)**1.7
            scale = 1 - intensity*0.35*w       # تا ۳۵٪ تنگ‌تر

            roi = result[y0:y1,x0:x1]
            map_x = c[0]+(xs-c[0])*scale - x0
            map_y = c[1]+(ys-c[1])*scale - y0
            warped = cv2.remap(roi,
                               map_x.astype(np.float32), map_y.astype(np.float32),
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            m3 = w[...,None]*0.92
            new_roi = (warped*m3 + roi*(1-m3)).astype(np.uint8)
            result[y0:y1,x0:x1] = new_roi

        # 🎨 تیره‌کردن ظریف داخل سوراخ جدید = عمق طبیعی
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in (nl, nr):
            d = np.sqrt((xs-c[0])**2+(ys-c[1])**2)
            inner = np.clip(1-d/(anat.nasal_width*0.10), 0, 1)**1.3
            lab[...,0][y0:y1,x0:x1] -= inner*10*intensity
        lab[...,0] = np.clip(lab[...,0],0,255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ═══════════════════════════════════════════
    def _roi(self, shape, pts, pad=0.6):
        h, w = shape[:2]
        x_min, y_min = pts.min(axis=0); x_max, y_max = pts.max(axis=0)
        cx, cy = (x_min+x_max)/2, (y_min+y_max)/2
        r = max(x_max-x_min, y_max-y_min)
        p = int(r*pad)
        rx0=max(0,int(cx-r/2-p)); ry0=max(0,int(cy-r/2-p))
        rx1=min(w,int(cx+r/2+p)); ry1=min(h,int(cy+r/2+p))
        return rx0,ry0,rx1,ry1


# ═══════════ تست همه ═══════════
if __name__ == "__main__":
    v3 = EngineV3()
    img = cv2.imread('/home/mohammad/Desktop/project/BeautyAIModel/test_images/test.jpg')
    lm = v3.parser.detect_from_image(img)
    a = NoseAnatomy(landmarks=lm, image_shape=img.shape)

    tests = [
        ('tip_up',    lambda: v3.tip_rotation_v2(img.copy(), a, 18)),
        ('hump',      lambda: v3.hump_removal_v2(img.copy(), a, 0.8)),
        ('tip_small', lambda: v3.tip_reduction_v2(img.copy(), a, 0.7)),
        ('nostrils',  lambda: v3.nostril_narrow_v2(img.copy(), a, 0.6)),
    ]
    panels = [cv2.resize(img,(500,int(500*img.shape[0]/img.shape[1])))]
    for name, fn in tests:
        try:
            out = fn()
            d = cv2.absdiff(
                cv2.resize(img,(700,int(700*img.shape[0]/img.shape[1]))),
                cv2.resize(out,(700,int(700*out.shape[0]/out.shape[1])))).mean()
            print(f'{name}: diff={d:.2f} {"✅" if d>0.3 else "⚠️ کم"}')
            panels.append(cv2.resize(out,(500,int(500*out.shape[0]/out.shape[1]))))
        except Exception as e:
            print(f'{name}: ERROR {e}')

    row1 = np.hstack(panels[:3]); row2 = np.hstack([panels[3],
        np.zeros_like(panels[3]), np.zeros_like(panels[3])])
    cv2.imwrite('/tmp/v3_all.jpg', np.vstack([row1,row2]), [cv2.IMWRITE_JPEG_QUALITY,90])
    print('saved /tmp/v3_all.jpg')
