"""
⚡ موتور هیبرید — بهترینِ هر دو دنیا
====================================
  تغییر ملایم + فرانت → موتور آناتومیک (سریع، ~6s)
  نیم‌رخ یا شدت بالا   → مدل مولد GPU (واقع‌گرا، ~15s)
  بدون GPU            → fallback هوشمند به آناتومیک با تنظیمات تقویتی

این فایل در three_d.py قبل از dispatch صدا زده میشود.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# تشخیص دسترس‌پذیری مدل مولد — یکبار
_GEN_AVAILABLE: Optional[bool] = None


def generative_available() -> bool:
    global _GEN_AVAILABLE
    if _GEN_AVAILABLE is None:
        try:
            from app.services.generative import is_available
            _GEN_AVAILABLE = bool(is_available())
        except Exception as e:
            logger.info(f"generative engine not available: {e}")
            _GEN_AVAILABLE = False
    return _GEN_AVAILABLE


def should_use_generative(view: str, intensity: float, action: str) -> tuple[bool, str]:
    """تصمیم مسیر پردازش بر اساس نوع درخواست"""
    heavy_actions = {
        'upturned_tip', 'droopy_tip', 'hump_reduction',
        'fantasy', 'doll_tip', 'ideal_realistic',
    }
    # نیم‌رخ همیشه سخت است
    if view in ('left_profile', 'right_profile'):
        return True, "profile view"
    # شدت بالا
    if intensity >= 0.75:
        return True, "high intensity"
    # اکشن‌های سنگین
    if action in heavy_actions and intensity >= 0.55:
        return True, f"heavy action {action}"
    return False, "mild front edit"


STYLE_PROMPTS = {
    'narrower':      "slimmer narrower nose",
    'wider':         "slightly wider nose base",
    'upturned_tip':  "upturned lifted nasal tip",
    'droopy_tip':    "downward angled nasal tip",
    'doll_tip':      "small cute doll-like rounded nose tip",
    'fantasy':       "elegant refined fantasy nose shape",
    'half_fantasy':  "subtly refined nose shape",
    'hump_reduction': "smooth straight nose bridge without hump",
    'smaller':       "proportionally smaller nose",
    'natural':       "very subtle natural nose refinement",
    'ideal_realistic': "ideal harmonious nose matching face proportions",
    'filler':        "non-surgical nose bridge filler result",
    'slim_bridge':   "thin slim defined nose bridge",
    'fleshy':        "softer fuller nose tip",
    'bony':          "defined bony nose structure",
    'shorter':       "shorter nasal length",
    'longer':        "slightly longer nasal length",
}


def generate_edit(image_bgr, action: str, intensity: float):
    """اجرا از طریق مدل مولد — فقط وقتی GPU باشد"""
    from app.services.generative import GenerativeEngine
    engine = GenerativeEngine.get()
    if engine is None:
        raise RuntimeError("generative unavailable")
    prompt = STYLE_PROMPTS.get(action, 'refined natural nose')
    return engine.edit_nose(image_bgr, prompt, intensity)
