"""
معیارهای زیبایی‌شناسی صورت (Clinical Aesthetic Standards)
=========================================================
منابع: تحلیل‌های کلاسیک رینوپلاستی و ارتوگناتیک که استاندارد بالینی هستند:
  • Steiner cephalometric analysis (SNA/SNB, nasolabial angle)
  • Ricketts aesthetic triangle
  • Farkas neoclassical canons (فاصله چشم‌ها = عرض بینی، ...)
  • Powell & Humphreys aesthetic triangle
  • نسبت طلایی φ = 1.618

این ماژول فقط «سقف مجاز» تغییرات را تعیین می‌کند تا خروجی همیشه
داخل محدوده زیبایی‌شناختی بماند و عکس «خراب» نشود.
"""
from typing import Optional

# =========================================================
#   بینی — مقادیر مرجع (درجات)
# =========================================================
NOSE_CANON = {
    # زاویه nasofrontal (رادیکس ↔ خط پیشانی): طبیعی 115–130
    'nasofrontal_angle': {'min': 115, 'max': 130, 'ideal_female': 134, 'ideal_male': 122},
    # زاویه nasolabial (نوک ↔ لب بالا): خانم 95–110 ، آقا 90–95
    'nasolabial_angle': {'female': (95, 110), 'male': (90, 95)},
    # زاویه nasofacial: 30–40
    'nasofacial_angle': {'min': 30, 'max': 40},
    # چرخش نوک (tip rotation): خانم 100–120 از خط فیلترال
    'tip_rotation_deg': {'female': (100, 120), 'male': (90, 105)},
    # عرض بینی ≈ فاصله اینترکانتال (Farkas canon) → سقف پهنای مجاز
    'width_to_intercanthal': 1.0,
    # حداکثر جابجایی نوک بر حسب ارتفاع بینی (جلوگیری از تخریب تصویر)
    'max_tip_lift_frac': 0.16,
    'max_projection_change_frac': 0.12,
    'max_width_change_frac': 0.28,
    'max_height_change_frac': 0.20,
}

# =========================================================
#   لب — مقادیر مرجع
# =========================================================
LIP_CANON = {
    # نسبت لب بالا به پایین: 1 : 1.6 (golden ratio)
    'upper_lower_ratio': 1.6,
    # عرض لب ≈ 0.39 × عرض صورت (نئوکلاسیک)
    'mouth_to_face_width': 0.39,
    # حداکثر بیرون‌زدگی هر لب (fraction طول لب)
    'max_volume_frac': 0.20,
    'max_corner_lift_frac': 0.10,
}

# =========================================================
#   سقف شدت هر استایل — مهم‌ترین لایه ضد-تخریب
#   (کاربر ممکن است «خیلی زیاد» بگوید؛ خروجی باز هم طبیعی می‌ماند)
# =========================================================
STYLE_INTENSITY_CAPS = {
    # بینی
    'smaller': 0.85, 'bigger': 0.80, 'narrower': 0.85, 'wider': 0.80,
    'shorter': 0.80, 'longer': 0.75,
    'upturned_tip': 0.75,      # بیش از این نوک «سوراخ» دیده می‌شود
    'doll_tip': 0.70,
    'fleshy': 0.70,
    'bony': 0.85,
    'fantasy': 0.80,
    'half_fantasy': 0.80,
    'natural': 0.60,
    'ideal_realistic': 0.70,
    'filler': 0.65,
    'slim_bridge': 0.85,
    # لب
    'fuller': 0.80, 'thinner': 0.75,
    'russian': 0.75, 'brazilian': 0.75, 'hollywood': 0.75,
    'heart_shape': 0.75, 'classic': 0.70, 'natural_lip': 0.60,
    'cupids_bow': 0.70, 'corner_lift': 0.70,
}


def clamp_intensity(action, intensity: float = 0.5) -> float:
    """شدت را به سقف زیبایی‌شناختی محدود می‌کند.
    امضای انعطاف‌پذیر: clamp_intensity(0.7) یا clamp_intensity('smaller', 0.7)
    """
    act: Optional[str] = action if isinstance(action, str) else None
    if act is None:
        intensity = float(action)
    cap = STYLE_INTENSITY_CAPS.get(act or '', 0.9)
    i = max(0.05, min(float(intensity), 1.0))
    return round(min(i, cap), 3)


def effective_geometry_caps(style: str) -> dict:
    """حد جابجایی/مقیاس مجاز برای یک استایل — برای موتور warp."""
    c = STYLE_INTENSITY_CAPS.get(style, 0.8)
    return {
        'tip_lift': NOSE_CANON['max_tip_lift_frac'] * c,
        'projection': NOSE_CANON['max_projection_change_frac'] * c,
        'width': NOSE_CANON['max_width_change_frac'] * c,
        'height': NOSE_CANON['max_height_change_frac'] * c,
    }
