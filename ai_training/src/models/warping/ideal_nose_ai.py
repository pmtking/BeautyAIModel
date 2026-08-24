"""
موتور هوشمند «بهترین حالت بینی» — Ultra-Realistic Ideal Nose AI
================================================================
برخلاف استایل‌های ثابت، این موتور مثل یک مدل هوش مصنوعی واقعی عمل می‌کند:

  ۱) تحلیل (Measure)   — اندازه‌گیری کلینیکی بینی و صورت از لندمارک‌های
                          MediaPipe بر اساس کانون‌های نئوکلاسیک (Farkas,
                          Powell & Humphreys, Steiner)
  ۲) برنامه (Plan)     — ساخت برنامه اصلاح «شخصی‌سازی‌شده» برای همین چهره؛
                          هر مانور فقط به‌اندازهٔ «کسری» همان شاخص اعمال میشود
  ۳) اجرا (Execute)    — اعمال مانورهای آناتومیک روی تصویر
  ۴) راستی‌آزمایی (Verify) — اندازه‌گیری مجدد روی خروجی و در صورت نیاز
                          اصلاح پسماند (حلقه بازخورد، حداکثر ۲ پاس)

اصل واقع‌گرایی:
  • هیچ تغییری فراتر از سقف‌های NOSE_CANON انجام نمی‌شود
  • تقارن ۱۰۰٪ مصنوعی است؛ عدم‌تقارن جزئی چهره عمداً حفظ می‌شود
  • هدف رسیدن به «نسبت ایده‌آل» است نه بیشینه‌سازی زیبایی
"""
import math
import logging
import numpy as np
from typing import Dict, List, Optional, Callable

try:
    from ...beauty_standards import clamp_intensity, NOSE_CANON
    from .nose_anatomy import NoseAnatomy
except Exception:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, os.path.dirname(__file__))
    from beauty_standards import clamp_intensity, NOSE_CANON
    from nose_anatomy import NoseAnatomy

from .nose_styles import Maneuvers, _resolve

logger = logging.getLogger(__name__)


# ============================================================
#   ۱) تحلیل‌گر — اندازه‌گیری‌های کلینیکی
# ============================================================

class IdealNoseAnalyzer:
    """اندازه‌گیری نسبت‌های کلیدی بینی در نمای روبرو."""

    # آستانه‌های قابل‌قبول (خارج از این بازه‌ها اصلاح لازم است)
    WIDTH_RATIO_OK = (0.92, 1.08)      # عرض بینی / فاصله اینترکانتال ≈ 1
    WH_RATIO_IDEAL = (1.55, 1.80)      # ارتفاع / عرض بینی
    TILT_DEG_IDEAL = (4.0, 14.0)       # انحراف محور نوک از عمود (چرخش طبیعی)
    ASYM_TOL = 0.06                    # عدم‌تقارن مجاز سوراخ‌ها (۶٪)

    @classmethod
    def analyze(cls, anat: NoseAnatomy, face: Dict) -> Dict:
        m: Dict = {'valid': bool(anat.valid)}
        if not anat.valid:
            return m

        nw, nh = anat.nasal_width, anat.nasal_height
        intercanthal = float(face.get('intercanthal', nw))

        # ---- عرض نسبت به کانون اینترکانتال (Farkas) ----
        wr = nw / max(intercanthal, 1e-6)
        m['width_ratio'] = round(wr, 3)
        lo, hi = cls.WIDTH_RATIO_OK
        # کسری عرض: فقط مقداری که از بازه مجاز بیرون است
        m['width_excess'] = round(max(0.0, wr - hi) / max(wr, 1e-6), 3)

        # ---- تناسب ارتفاع به عرض ----
        wh = nh / max(nw, 1e-6)
        m['wh_ratio'] = round(wh, 3)
        ilo, ihi = cls.WH_RATIO_IDEAL
        m['height_deficit'] = 0.0
        m['height_excess'] = 0.0
        if wh < ilo:
            m['height_deficit'] = round((ilo - wh) / ilo, 3)   # بینی کوتاه؟
        elif wh > ihi:
            m['height_excess'] = round((wh - ihi) / ihi, 3)    # کشیده؟

        # ---- زاویه چرخش نوک (تقریب ۲بعدی از خط رادیکس→نوک) ----
        tip = anat.get('tip'); radix = anat.get('radix')
        tilt_deg = 0.0
        if tip is not None and radix is not None:
            v = tip - radix
            # عمودِ مرجع: خط عمودی گذرا از مرکز پایه (ضد-کجی مثل Maneuvers)
            vx = float(anat.alar_mid[0] - radix[0])
            vy = float(tip[1] - radix[1])
            tilt_deg = math.degrees(math.atan2(abs(vx), max(abs(vy), 1e-6)))
        m['tilt_deg'] = round(float(tilt_deg), 1)
        tlo, thi = cls.TILT_DEG_IDEAL
        m['tilt_deficit'] = round(max(0.0, tlo - tilt_deg) / 90.0, 4)
        m['tilt_excess'] = round(max(0.0, tilt_deg - thi) / 90.0, 4)

        # ---- عدم‌تقارن سوراخ‌ها ----
        nl, nr = anat.get('nostril_l'), anat.get('nostril_r')
        base = anat.get('base_center')
        asym = 0.0
        if nl is not None and nr is not None and base is not None:
            dl = float(np.linalg.norm(base - nl))
            dr = float(np.linalg.norm(base - nr))
            asym = abs(dl - dr) / max((dl + dr) / 2, 1e-6)
        m['nostril_asym'] = round(asym, 3)
        m['needs_symmetry'] = bool(asym > cls.ASYM_TOL)

        # ---- امتیاز کیفیت کلی (0..100) ----
        score = 100.0
        score -= min(30.0, m['width_excess'] * 300)          # عرض
        score -= min(20.0, m['height_excess'] * 200)         # کشیدگی
        score -= min(10.0, m['height_deficit'] * 200)        # کوتاهی
        score -= min(20.0, m['tilt_excess'] * 900)           # نوک افتاده
        score -= min(10.0, m['tilt_deficit'] * 900)          # نوک بیش‌حد بالا
        score -= min(10.0, max(0.0, asym - cls.ASYM_TOL) * 250)
        m['quality_score'] = int(round(max(0.0, score)))
        m['warnings'] = []
        if wr > 1.15:
            m['warnings'].append('عرض بینی заметناً بیشتر از فاصله اینترکانتال است')
        if tilt_deg > 20:
            m['warnings'].append('نوک بینی افتادگی دارد')
        if tilt_deg < 2:
            m['warnings'].append('چرخش نوک بیش از حد طبیعی است (نمای عمل‌شده)')
        return m


# ============================================================
#   ۲) برنامه‌ساز — شخصی‌سازی‌شده برای همین چهره
# ============================================================

class IdealNosePlanner:

    @staticmethod
    def build(analysis: Dict, anat: NoseAnatomy,
              user_intensity: float) -> List[Dict]:
        """هر مانور = {maneuver, amount} — مقادیر از کسری شاخص‌ها می‌آیند."""
        i = float(np.clip(user_intensity, 0.05, 1.0))
        nw, nh = anat.nasal_width, anat.nasal_height
        plan: List[Dict] = []

        # ---- عرض بال‌ها: فقط به‌اندازهٔ کسری که تحلیل گفت ----
        w_excess = analysis.get('width_excess', 0.0)
        if w_excess > 0.01:
            # 🎯 قوی اما امن: تا ۲۸٪ عرض؛ ضریب ۱.۳ برای جبران نرمی پنجره کسینوسی
            amount = min(nw * 0.28, nw * w_excess * 1.3) * i
            plan.append({'maneuver': 'alar_narrowing', 'amount': float(amount),
                         'reason': f"عرض {analysis['width_ratio']}× اینترکانتال → جمع‌کردن بال‌ها"})

        # ---- قوس پل: صاف‌سازی ملایم (همیشه کم — جلوگیری از نمای تخت) ----
        plan.append({'maneuver': 'dorsum_reshape', 'amount': -0.06 * i,
                     'reason': 'صاف‌سازی خیلی ملایم خط مرکزی پل'})

        # ---- نوک: فقط اگر خارج از بازه طبیعی باشد ----
        tilt_def = analysis.get('tilt_deficit', 0.0)
        tilt_exc = analysis.get('tilt_excess', 0.0)
        if tilt_def > 0.005:
            deg = min(7.0, 90.0 * tilt_def) * i               # حداکثر ۷°
            plan.append({'maneuver': 'tip_rotation', 'amount': float(deg),
                         'reason': f"چرخش نوک به بالا ({deg:.1f}°) — نوک کمی افتاده"})
        elif tilt_exc > 0.005:
            deg = min(4.0, 90.0 * tilt_exc) * i               # برگشت خیلی ملایم
            plan.append({'maneuver': 'tip_rotation', 'amount': float(-deg),
                         'reason': f'چرخش بیش از حد طبیعی → کاهش {deg:.1f} درجه‌ای'})

        # ---- پروجکشن نوک: بینی بیش‌حد بلند → کمی جمع‌کردن نوک ----
        h_exc = analysis.get('height_excess', 0.0)
        if h_exc > 0.03:
            amt = min(nh * 0.05, nh * h_exc * 0.4) * i
            plan.append({'maneuver': 'columella_set', 'amount': float(-amt),
                         'reason': 'کاهش ظریف طول با جمع‌کردن کولوملا'})

        # ---- تقارن سوراخ‌ها: فقط وقتی عدم‌تقارن محسوس است ----
        if analysis.get('needs_symmetry'):
            plan.append({'maneuver': 'nostril_symmetry', 'amount': 0.5 * i,
                         'reason': f"عدم‌تقارن سوراخ‌ها {int(analysis['nostril_asym']*100)}٪"})

        # ---- تعریف دیواره‌ها: همیشه ظریف (عمق واقعی، نه تخت مصنوعی) ----
        plan.append({'maneuver': 'sidewall_definition',
                     'amount': float(nw * 0.04 * i),
                     'reason': 'تعریف ظریف دیواره‌ها برای عمق طبیعی'})
        return plan


# ============================================================
#   ۳) اجرا + راستی‌آزمایی (حلقه بازخورد)
# ============================================================

class IdealNoseAI:
    """خط لوله کامل: تحلیل → برنامه → اجرا → اندازه‌گیری مجدد → اصلاح پسماند."""

    MAX_PASSES = 2          # پاس اصلی + حداکثر یک اصلاح پسماند
    RESIDUAL_GAIN = 0.65    # ضریب اصلاح دوم (ملایم تا over-shoot نشود)

    def __init__(self):
        self.last_report: Optional[Dict] = None

    # ---------- زمینه صورت ----------
    @staticmethod
    def _face_context(landmarks, shape) -> Dict:
        """استخراج فاصله اینترکانتال و سایر مراجعه صورت (پیکسل)."""
        try:
            if not landmarks or len(landmarks) < 468 or shape is None:
                return {}
            h, w = shape[:2]
            first = landmarks[0]
            norm = (0.0 <= float(first['x']) <= 1.0 and
                    0.0 <= float(first['y']) <= 1.0)
            s = np.array([w, h], dtype=np.float32) if norm \
                else np.array([1.0, 1.0], dtype=np.float32)

            def P(idx):
                lm = landmarks[idx]
                return np.array([float(lm['x']), float(lm['y'])],
                                dtype=np.float32) * s

            intercanthal = float(np.linalg.norm(P(133) - P(362)))  # گوشه داخلی چشم‌ها
            return {
                'intercanthal': intercanthal,
                'eye_width_l': float(np.linalg.norm(P(33) - P(133))),
                'eye_width_r': float(np.linalg.norm(P(263) - P(362))),
                'face_width': float(np.linalg.norm(P(234) - P(454))),
            }
        except Exception as e:
            logger.debug(f'face_context failed: {e}')
            return {}

    # ---------- اجرای برنامه ----------
    @staticmethod
    def _execute(image, anat, plan) -> np.ndarray:
        result = image
        for step in plan:
            fn = getattr(Maneuvers, step['maneuver'], None)
            if fn is None:
                continue
            try:
                result = fn(result, anat, step['amount'])
            except Exception as e:
                logger.warning(f'maneuver {step["maneuver"]} failed: {e}')
        return result

    # ---------- خط لوله اصلی ----------
    def transform(self, image: np.ndarray, landmarks, shape,
                  intensity: float = 0.65,
                  detector: Optional[Callable] = None) -> np.ndarray:
        """
        detector: تابع (image)->landmarks برای اندازه‌گیری مجدد روی خروجی
                  (در صورت نبود، فقط یک پاس اجرا می‌شود).
        گزارش کامل در self.last_report ذخیره می‌شود.
        """
        report: Dict = {'passes': [], 'mode': 'ai-personalized'}
        a = _resolve(landmarks, shape)
        if a is None:
            report['error'] = 'anatomy not resolved'
            self.last_report = report
            return image

        face = self._face_context(landmarks, shape)
        analysis = IdealNoseAnalyzer.analyze(a, face)
        report['before'] = analysis
        if not analysis.get('valid'):
            self.last_report = report
            return image

        plan = IdealNosePlanner.build(analysis, a, intensity)
        report['plan'] = [{'maneuver': s['maneuver'],
                           'amount': round(s['amount'], 2),
                           'reason': s['reason']} for s in plan]

        result = self._execute(image, a, plan)
        report['passes'].append({
            'pass': 1, 'steps': len(plan),
            'score_after': analysis.get('quality_score'),
        })

        # ---------- حلقه راستی‌آزمایی ----------
        if detector is not None and plan:
            try:
                lm2 = detector(result)
                if lm2:
                    a2 = _resolve(lm2, shape)
                    if a2 is not None and a2.valid:
                        face2 = self._face_context(lm2, shape)
                        an2 = IdealNoseAnalyzer.analyze(a2, face2)
                        improved = an2.get('quality_score', 0) >= \
                            analysis.get('quality_score', 0)
                        # اصلاح پسماند فقط اگر هنوز کسری معنادار مانده
                        residual = max(an2.get('width_excess', 0.0),
                                       an2.get('tilt_deficit', 0.0) * 3)
                        if improved and residual > 0.02 and \
                                self.MAX_PASSES > 1:
                            plan2 = IdealNosePlanner.build(
                                an2, a2, intensity * self.RESIDUAL_GAIN)
                            if plan2:
                                result = self._execute(result, a2, plan2)
                                report['passes'].append({
                                    'pass': 2, 'steps': len(plan2),
                                    'residual': round(residual, 3),
                                    'score_before': an2.get('quality_score'),
                                })
                                # اندازه‌گیری نهایی برای گزارش
                                lm3 = detector(result)
                                if lm3:
                                    a3 = _resolve(lm3, shape)
                                    if a3 is not None and a3.valid:
                                        an3 = IdealNoseAnalyzer.analyze(
                                            a3, self._face_context(lm3, shape))
                                        report['after'] = {
                                            'quality_score':
                                                an3.get('quality_score'),
                                            'width_ratio':
                                                an3.get('width_ratio'),
                                            'tilt_deg': an3.get('tilt_deg'),
                                        }
                            else:
                                report['after'] = {
                                    'quality_score': an2.get('quality_score')}
                        else:
                            report['after'] = {
                                'quality_score': an2.get('quality_score'),
                                'early_stop': not improved}
            except Exception as e:
                logger.warning(f'verify pass skipped: {e}')

        report['status'] = 'ok'
        self.last_report = report
        return result


# سینگلتون سراسری — گزارش آخر برای API
IDEAL_NOSE_AI = IdealNoseAI()
