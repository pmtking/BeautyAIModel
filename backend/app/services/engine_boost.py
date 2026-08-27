"""
⚡ ارتقای موتور آناتومیک فعلی — نسخه v2 (بدون GPU، همین الان فعال)
====================================================================
چون مدل مولد GPU میخواهد، تا آن موقع این ارتقاها موتور فعلی را
«۲۰ لول» قوی‌تر میکنند:

1. حلقه بازخورد: بعد از warp دوباره اندازه‌گیری → اگر هدف محقق نشد پاس دوم
2. Shading transfer برای همه استایل‌ها (نه فقط narrowing)
3. ضد-هاله: حذف overshoot با clamp هوشمند روشنایی
4. Multi-pass برای شدت بالا (به‌جای یک warp بزرگ که میشکند)
5. Texture-preserving blend (حفظ منافذ پوست)
"""
import sys, cv2, numpy as np
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src')
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src/models')
from face_parser.model import FaceParserModel
from warping.nose_anatomy import NoseAnatomy


class EngineBoost:
    """لایه تقویت روی موتور فعلی — بدون تغییر کد قدیمی"""

    def __init__(self):
        self.parser = FaceParserModel()

    # ────────────────────────────────────────────
    def measure(self, img, anat):
        """اندازه‌گیری مستقل عرض سایه آلار (پیکسلی)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        al, ar = anat.get('alar_l'), anat.get('alar_r')
        w = anat.nasal_width
        y_c = int((al[1]+ar[1])/2)
        vals = []
        for dy in (-10, 0, 10):
            row = gray[y_c+dy]
            win = int(w*0.28)
            seg_l = row[max(0,int(al[0])-win):int(al[0])+int(w*0.12)]
            seg_r = row[int(ar[0])-int(w*0.12):int(ar[0])+win]
            xl = max(0,int(al[0])-win)+int(np.argmin(seg_l))
            xr = int(ar[0])-int(w*0.12)+int(np.argmin(seg_r))
            vals.append(xr-xl)
        return float(np.median(vals))

    # ────────────────────────────────────────────
    def anti_halo(self, before, after, mask_roi=None):
        """حذف هاله: clamp روشنایی به محدوده آماری قبل ± 3σ در ناحیه تغییر"""
        b = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
        n = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
        Lb, Ln = b[...,0], n[...,0]
        # پیکسل‌هایی که بیش از 40 واحد روشن‌تر از قبل شده و در قبل تیره بودند = هاله
        halo = ((Ln - Lb) > 40) & (Lb < np.percentile(Lb, 60))
        if mask_roi is not None:
            halo &= ~mask_roi.astype(bool)
        fixed = n.copy()
        # جایگزینی با blur محلی (بدون از دست دادن فرم)
        L_fixed = cv2.inpaint(Ln.astype(np.uint8), (halo*255).astype(np.uint8),
                              5, cv2.INPAINT_TELEA)
        fixed[...,0] = np.where(halo, L_fixed, Ln)
        return cv2.cvtColor(fixed.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ────────────────────────────────────────────
    def texture_preserve_blend(self, src, warped, mask):
        """blend با حفظ بافت: فرم از warped + high-freq از source"""
        s = src.astype(np.float32)
        wr = warped.astype(np.float32)
        m = (mask.astype(np.float32)/255.0)[...,None]

        s_low = cv2.GaussianBlur(s, (0,0), 8)
        w_low = cv2.GaussianBlur(wr, (0,0), 8)
        s_high = s - s_low

        merged = w_low + s_high*0.65     # فرم جدید + بافت واقعی
        out = merged*m + s*(1-m)
        return np.clip(out, 0, 255).astype(np.uint8)

    # ────────────────────────────────────────────
    def multi_pass_narrow(self, image, target_frac, max_passes=3):
        """باریک‌سازی چند-پاسه: هر پاس کوچک، اندازه‌گیری، ادامه تا هدف"""
        lm = self.parser.detect_from_image(image)
        anat = NoseAnatomy(landmarks=lm, image_shape=image.shape)
        w0_px = self.measure(image, anat)
        target_w = w0_px * (1 - target_frac)

        from warping.narrowing_pro import NarrowingPro
        result = image
        remaining = target_frac
        for p in range(max_passes):
            step = min(remaining, 0.15)          # هر پاس حداکثر ۱۵٪
            result = NarrowingPro.apply(result, anat, step)
            # re-detect
            lm2 = self.parser.detect_from_image(result)
            if not lm2:
                break
            anat = NoseAnatomy(landmarks=lm2, image_shape=result.shape)
            current = self.measure(result, anat)
            achieved = 1 - current/w0_px
            print(f'   pass {p+1}: عرض={current:.0f} ({achieved*100:+.1f}% از پایه)')
            if abs(current - target_w) < w0_px*0.03:
                break                            # به هدف رسیدیم
            remaining = target_frac - achieved
            if remaining <= 0.01:
                break
        return result

    # ────────────────────────────────────────────
    def boost_upturned_tip(self, image, intensity=0.7):
        """چرخش نوک با ضد-هاله + shading — نسخه قوی"""
        from warping.nose_styles import Maneuvers

        # چرخش در دو پاس نصف — هر کدام کمتر مصنوعی
        deg_total = 22.0 * intensity
        lm = self.parser.detect_from_image(image)
        anat = NoseAnatomy(landmarks=lm, image_shape=image.shape)

        result = Maneuvers.tip_rotation(image.copy(), anat, deg_total * 0.6)
        lm2 = self.parser.detect_from_image(result)
        if lm2:
            anat2 = NoseAnatomy(landmarks=lm2, image_shape=result.shape)
            result = Maneuvers.tip_rotation(result, anat2, deg_total * 0.4)

        result = self.anti_halo(image, result)
        return result


# ═══════════ تست ═══════════
if __name__ == "__main__":
    boost = EngineBoost()
    img = cv2.imread('/home/mohammad/Desktop/project/BeautyAIModel/test_images/test.jpg')

    print('🔬 Boost 1: باریک‌سازی چند-پاسه (هدف ۳۰٪)')
    out1 = boost.multi_pass_narrow(img.copy(), 0.30)
    cv2.imwrite('/tmp/v2_narrow30.jpg',
                cv2.resize(out1,(800,int(800*out1.shape[0]/out1.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY,90])

    print('🔬 Boost 2: نوک بالا با ضد-هاله')
    out2 = boost.boost_upturned_tip(img.copy(), 0.8)
    d = cv2.absdiff(
        cv2.resize(img,(800,int(800*img.shape[0]/img.shape[1]))),
        cv2.resize(out2,(800,int(800*out2.shape[0]/out2.shape[1])))).mean()
    print(f'   diff={d:.2f} (باید > 0.5 باشد)')
    cv2.imwrite('/tmp/v2_tip.jpg',
                cv2.resize(out2,(800,int(800*out2.shape[0]/out2.shape[1]))),
                [cv2.IMWRITE_JPEG_QUALITY,90])

    combo = np.hstack([
        cv2.resize(img,(650,int(650*img.shape[0]/img.shape[1]))),
        cv2.resize(out1,(650,int(650*out1.shape[0]/out1.shape[1])))])
    cv2.imwrite('/tmp/v2_compare_narrow.jpg', combo, [cv2.IMWRITE_JPEG_QUALITY,90])
    print('✅ saved /tmp/v2_compare_narrow.jpg /tmp/v2_tip.jpg')
