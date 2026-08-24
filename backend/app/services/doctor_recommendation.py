# app/services/doctor_recommendation.py
"""
سیستم پیشنهاد پزشک و برآورد درمان.
پس از ساخت مدل سه‌بعدی و اعمال تغییرات، این سرویس:
  1. ناحیه/نوع تغییر را به خدمت کلینیکی نگاشت می‌کند
  2. بازه قیمت تقریبی (تومان) و تعداد جلسات را تخمین می‌زند
  3. بهترین پزشکِ مرتبط از پنل پزشکان را پیشنهاد می‌دهد
  4. پیام شخصی‌سازی‌شده برای CTA رزرو مشاوره تولید می‌کند
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# =========================================================
#   کاتالوگ خدمات — قیمت‌ها به تومان (بازه تقریبی بازار)
# =========================================================

SERVICE_CATALOG: Dict[str, Dict] = {
    'lip': {
        'title': 'افزایش حجم لب (فیلر لب)',
        'price_per_cc_toman': 4_500_000,
        'sessions': 1,
        'duration_min': 30,
        'recovery_days': 3,
        'specialties': ['پوست', 'زیبایی'],
        'note': 'فیلر هیالورونیک اسید با دوام ۸ تا ۱۲ ماه',
    },
    'nose': {
        'title': 'جراحی زیبایی بینی (رینوپلاستی)',
        'price_range_toman': (120_000_000, 250_000_000),
        'sessions': 1,
        'duration_min': 180,
        'recovery_days': 21,
        'specialties': ['گوش، حلق و بینی'],
        'note': 'دوره کبودی و تورم حدود ۲ تا ۳ هفته',
    },
    'jaw': {
        'title': 'فرم‌دهی فک و چانه (فیلر فک یا پروتز)',
        'price_per_cc_toman': 5_000_000,
        'price_range_toman': (8_000_000, 45_000_000),
        'sessions': 1,
        'duration_min': 45,
        'recovery_days': 5,
        'specialties': ['پوست', 'زیبایی', 'فک و صورت'],
        'note': 'بسته به پروتز یا فیلر، نتیجه موقت یا دائمی',
    },
    'cheek': {
        'title': 'پرکردن گونه (فیلر گونه)',
        'price_per_cc_toman': 5_000_000,
        'price_range_toman': (9_000_000, 30_000_000),
        'sessions': 1,
        'duration_min': 40,
        'recovery_days': 3,
        'specialties': ['پوست', 'زیبایی'],
        'note': 'لیفت طبیعی صورت با حجم‌دهی گونه',
    },
    'eye': {
        'title': 'جوان‌سازی اطراف چشم (فیلر زیر چشم)',
        'price_range_toman': (12_000_000, 35_000_000),
        'sessions': 1,
        'duration_min': 40,
        'recovery_days': 5,
        'specialties': ['پوست', 'زیبایی'],
        'note': 'رفع گودی و سیاهی زیر چشم',
    },
    'forehead': {
        'title': 'تزریق بوتاکس پیشانی',
        'unit_label': 'واحد',
        'price_range_toman': (4_000_000, 9_000_000),
        'sessions': 1,
        'duration_min': 20,
        'recovery_days': 1,
        'specialties': ['پوست', 'زیبایی'],
        'note': 'دوام ۴ تا ۶ ماه',
    },
}

# =========================================================
#   پنل پزشکان (فعلاً ثابت — بعداً از دیتابیس خوانده می‌شود)
# =========================================================

DOCTOR_PANEL: List[Dict] = [
    {
        'id': 'dr-taheri',
        'name': 'دکتر محمد طاهری',
        'specialty': 'متخصص پوست، مو و زیبایی',
        'sub_specialties': ['پوست', 'زیبایی'],
        'clinic': 'کلینیک BUTI',
        'rating': 4.9,
        'review_count': 312,
        'experience_years': 12,
        'avatar_url': None,
        'bio': 'متخصص تزریقات حجم‌دهی و جوان‌سازی با رویکرد طبیعی',
    },
    {
        'id': 'dr-sample-ent',
        'name': 'دکتر نمونه (گوش، حلق و بینی)',
        'specialty': 'جراح بینی و صورت',
        'sub_specialties': ['گوش، حلق و بینی', 'فک و صورت'],
        'clinic': 'کلینیک BUTI',
        'rating': 4.8,
        'review_count': 198,
        'experience_years': 15,
        'avatar_url': None,
        'bio': 'رینوپلاستی اولتراسونیک با ریکاوری سریع',
    },
]

# نگاشت action به فعل بازاریابی
ACTION_LABELS = {
    'fuller': 'حجم‌دهی',
    'bigger': 'افزایش حجم',
    'smaller': 'کوچک‌سازی',
    'sharper': 'زاویه‌دهی',
    'lift': 'لیفت',
}


def _estimate_price(service: Dict, cc: Optional[float]) -> Optional[Dict]:
    """بازه قیمت را بر اساس سرویس و مقدار ژل تخمین می‌زند."""
    price_per_cc = service.get('price_per_cc_toman')
    price_range = service.get('price_range_toman')

    if price_per_cc and cc:
        total = price_per_cc * cc
        return {
            'min': int(total * 0.9),
            'max': int(total * 1.15),
            'label': f'{cc:g} سی‌سی × {_fmt(price_per_cc)}',
        }
    if price_per_cc:
        return {'min': price_per_cc, 'max': int(price_per_cc * 1.3), 'label': 'هر سی‌سی'}
    if price_range:
        return {'min': price_range[0], 'max': price_range[1], 'label': 'بازه کلینیکی'}
    return None


def _fmt(n: int) -> str:
    """4500000 -> ۴.۵ میلیون (خوانا برای UI)"""
    if n >= 1_000_000:
        v = n / 1_000_000
        return f'{v:g} میلیون تومان'
    return f'{n:,} تومان'


def recommend_for_change(
    area: Optional[str],
    action: Optional[str],
    intensity: float,
    gel_cc: Optional[float] = None,
) -> Dict:
    """
    ورودی: خروجی changes بک‌اند + مقدار سی‌سی استخراج‌شده از متن
    خروجی: بسته کامل پیشنهاد برای نمایش در اپ
    """
    if not area or area not in SERVICE_CATALOG:
        return {}

    service = SERVICE_CATALOG[area]
    price = _estimate_price(service, gel_cc)

    # بهترین پزشک بر اساس تطابق تخصص
    needed_specs = set(service.get('specialties', []))
    ranked = sorted(
        DOCTOR_PANEL,
        key=lambda d: len(needed_specs & set(d.get('sub_specialties', []))),
        reverse=True,
    )
    doctor = ranked[0]

    action_fa = ACTION_LABELS.get(action or '', 'اصلاح')

    message = (
        f"بر اساس شبیه‌سازی شما، {service['title']} "
        f"با {action_fa} مناسب چهره‌تان است. "
    )
    if gel_cc:
        message += f"حدود {gel_cc:g} سی‌سی فیلر نیاز دارد. "

    return {
        'service': {
            'area': area,
            'title': service['title'],
            'note': service.get('note'),
            'sessions': service['sessions'],
            'duration_min': service['duration_min'],
            'recovery_days': service['recovery_days'],
        },
        'estimated_price': price and {
            **price,
            'currency': 'toman',
        },
        'gel_cc': gel_cc,
        'doctor': doctor,
        'message': message.strip(),
        'cta': {
            'text': 'رزرو مشاوره رایگان حضوری',
            'deeplink': f'buti://consult?doctor={doctor["id"]}&area={area}',
        },
    }


def extract_gel_cc(text: str) -> Optional[float]:
    """از متن کاربر مقدار سی‌سی/میلی را بیرون می‌کشد («سه سیسی ژل»).
    هم ارقام فارسی/عربی و هم اعداد حروفی (یک تا ده) را پشتیبانی می‌کند."""
    import re

    PERSIAN_DIGITS = str.maketrans('۰۱۲۳۴۵۶۷۸۹', '0123456789')
    ARABIC_DIGITS = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    t = text.lower().translate(PERSIAN_DIGITS).translate(ARABIC_DIGITS)

    # اعداد حروفی رایج
    WORD_NUMBERS = {
        'نیم': 0.5, 'یک': 1, 'دو': 2, 'سه': 3, 'چهار': 4, 'پنج': 5,
        'شش': 6, 'هفت': 7, 'هشت': 8, 'نه': 9, 'ده': 10,
    }

    # ۱) عدد رقمی: «3 cc»، «۲ سیسی»، «1.5 میلی»
    m = re.search(r'(\d+(?:[.,]\d+)?)\s*(cc|سی\s*سی|سیسی|ml|میلی)', t)
    if m:
        return min(max(float(m.group(1).replace(',', '.')), 0.25), 10.0)

    # ۲) عدد حروفی + واحد: «سه سیسی»، «دو میلی»
    for word, val in WORD_NUMBERS.items():
        if re.search(rf'{word}\s*(cc|سی\s*سی|سیسی|ml|میلی)', t):
            return float(val)

    # ۳) «ژل سه» یا «۳ تا ژل» بدون واحد صریح
    m = re.search(r'(\d+)\s*(تا\s*)?ژل', t)
    if m:
        return min(max(float(m.group(1)), 0.5), 10.0)
    for word, val in WORD_NUMBERS.items():
        if re.search(rf'{word}\s*(تا\s*)?ژل', t):
            return float(val)

    return None
