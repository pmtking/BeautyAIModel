"""
💬 موتور چت‌بات دوستانه Buti — «رفیق زیبایی»
=============================================
یک لایه NLU سبک و بدون وابستگی خارجی که:
  ۱. احساس پیام کاربر را می‌خواند (سؤال / تشکر / نگرانی / شوخی / سلام ...)
  ۲. متن دوستانه، کوتاه و شخصیت‌دار جواب می‌دهد (با ایموجی)
  ۳. اگر درخواست ادیت بود → آن را برای موتور پردازش استخراج می‌کند
  ۴. سابقه گفتگو را نگه می‌دارد تا جواب‌ها تکراری/بی‌ربط نشود

طراحی برای دموی سرمایه‌گذار: پاسخ‌های متنوع، لحن رفیقانه، صفر تأخیر شبکه.
"""

import re
import random
from typing import Dict, List, Optional

# ============================================
#  شخصیت — لحن رفیقانه
# ============================================

GREETINGS = [
    "سلام رفیق! 🙌 چه خبر؟ آماده‌ای یه تغییر جذاب ببینیم؟",
    "اِلا سلام! 😍 خوش اومدی — بگو امروز چی می‌خوای بشی!",
    "های! ✨ خوشحالم که هستی — از کجا شروع کنیم رفیق؟",
]

THANKS = [
    "فدات! 💛 هر وقت خواستی من همین‌جام.",
    "خواهش رفیق! 🌸 نظر خودت مهم‌تره — خوشت اومد؟",
    "کاری نکردم! 😄 تو که خودت خوشگلی.",
]

PRAISE_ACK = [
    "ایول! 🔥 پس بریم قشنگ‌ترش کنیم.",
    "مرسی رفیق! 🥰 حالا بگو چی رو عوض کنیم؟",
    "قربونت! ✨ منتظر دستورت هستم.",
]

WORRY_REPLIES = [
    "نگران نباش رفیق 🌷 هیچ‌چیز اینجا واقعی نیست — فقط یه شبیه‌سازی امنه که بدونی چه شکلی میشی. تصمیم نهایی همیشه با خودته و پزشک.",
    "کاملاً درکت می‌کنم 🤍 اینجا فقط پیش‌نمایشه؛ نه درد داره نه ریسک. با خیال راحت تست کن، اگه خوشت نیومد هیچی نشده!",
]

JOKES = [
    "😄 ما اینجاییم که خوشگل‌تر شی، نه اینکه غصه‌تو بخوریم! بگو بینی؟ لب؟ گونه؟",
    "😂 رفیق شوخی‌هاتو بعداً جواب میدم — الان اول کارمونه: یه سلفی بفرست!",
]

HELP = [
    "من رفیق زیباییتم! 💫 این کارها رو بلدم:\n"
    "• تحلیل چهره از روی سلفی 📸\n"
    "• شبیه‌سازی عمل بینی، لب، گونه، فک...\n"
    "• استایل‌های معروف: روسی، فانتزی، عروسکی ⭐\n"
    "• حدس هزینه و معرفی پزشک 👨‍⚕️\n"
    "فقط یه عکس بفرست و بگو چی می‌خوای!",
]

ASK_PHOTO = [
    "اوکی رفیق! 🙌 فقط یه سلفی یا عکس از گالری لازم دارم — بذار ببینیم قراره چی بشی! 😍",
    "اول یه عکس خوشگل از خودت بفرست 📸 بعدش شعبده‌بازی باهاته!",
]

NO_FACE_TIPS = [
    "اوپس! چهره‌تو نتونستم خوب ببینم 🙈\nیه سلفی روشن بگیر، صورت کامل داخل کادر باشه و عینک/ماسک رو بردار — بعد دوباره امتحان کن رفیق! 📸",
]

PROCESSING_LINES = [
    "دارم آنالیزت می‌کنم… 🔍✨",
    "چند ثانیه صبر کن، داری قشنگ‌تر میشی… 💫",
    "هوش مصنوعی داره رو چهره‌ت کار می‌کنه… 🪄",
    "در حال ساخت نسخه جدید تو… ⚡",
]

# پاسخ‌های بعد از موفقیت (متن + حس)
SUCCESS_LINES = [
    "تمومه! اینم از نتیجه ✨ چطوره؟ اگه دوستش نداری بگو دوباره بزنم 😉",
    "آماده شد رفیق! 🔥 نظر خودت چیه — شدتش رو کم/زیاد کنم؟",
    "اینم از تغییر! 🌟 قبل/بعد رو مقایسه کن — من که عاشقشم 😍",
]

# ============================================
#  تشخیص نیت
# ============================================

_PATTERNS = {
    'greeting': r'سلام|درود|hi|hello|های|صبح بخیر|شب بخیر',
    'thanks':   r'مرسی|ممنون|دمت گرم|تشکر|thanks|عالی بود|دستت درد نکنه',
    'praise':   r'خوبی|چطوری|چه خبر|رفیق|دوستت دارم|باحالی',
    'worry':    r'میترس|ترس|درد|دارد|خطر|عوارض|بی‌خطر|واقعیه|واقعی میشه|جراحی واقعی|برمیگرده|پشیمون',
    'joke':     r'شوخی|جوک|بخند|خنده|مسخره',
    'help':     r'چیکار|بلدی|کمک|راهنما|چه کاری|چی بلدی|help',
    'price':    r'قیمت|هزینه|چند|تومان|پول|نرخ|تعرفه',
    'doctor':   r'دکتر|پزشک|جراح|مشاوره|کلینیک|نوبت',
}

_EDIT_HINT = r'(کن|بشه|باشه|بزن|بده|ببر|بیار|کوچک|بزرگ|پر|باریک|پهن|تیز|بالا|پایین|روسی|برزیلی|فانتزی|عروسکی|گوشتی|استخوانی|قوز|قلمی|فیلر|طبیعی)'


def detect_intent(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return 'empty'
    for intent, pat in _PATTERNS.items():
        if re.search(pat, t):
            return intent
    if re.search(_EDIT_HINT, t):
        return 'edit'
    return 'chat'


# ============================================
#  چت‌بات
# ============================================

class BeautyChatBot:
    """حافظه سبک per-user؛ در FastAPI به‌صورت singleton استفاده شود."""

    def __init__(self):
        self._history: Dict[str, List[dict]] = {}
        self._last_replies: Dict[str, str] = {}

    # ---------- API اصلی ----------
    def reply(self, user_id: str, text: str,
              has_photo: bool = False, last_result_ok: Optional[bool] = None) -> dict:
        """
        خروجی:
          { reply, intent, is_edit_request }
        """
        intent = detect_intent(text)
        self._remember(user_id, {'role': 'user', 'text': text})

        if intent == 'edit':
            msg = None                      # موتور پردازش جواب می‌دهد
        elif intent == 'greeting':
            msg = self._pick(user_id, GREETINGS)
        elif intent == 'thanks':
            msg = self._pick(user_id, THANKS)
        elif intent == 'praise':
            msg = self._pick(user_id, PRAISE_ACK)
        elif intent == 'worry':
            msg = self._pick(user_id, WORRY_REPLIES)
        elif intent == 'joke':
            msg = self._pick(user_id, JOKES)
        elif intent == 'help':
            msg = HELP[0]
        elif intent == 'price':
            msg = ("قیمت دقیق به ناحیه و کلینیک بستگی داره 💰 ولی بعد از تحلیل عکس، "
                   "برآورد تقریبی و پزشک مناسب رو بهت معرفی می‌کنم!")
        elif intent == 'doctor':
            msg = ("برای مشاوره رایگان بهترین جراح‌ها رو دارم 👨‍⚕️ اول یه عکس بفرست تا "
                   "تحلیل کنم، بعد پزشک متخصص همون کار رو معرفی می‌کنم!")
        elif not has_photo and last_result_ok is None:
            msg = self._pick(user_id, ASK_PHOTO)
        else:
            msg = ("متوجه نشده رفیق 😅 یه بار دیگه بگو — مثلاً: «بینی عروسکی» یا «لب روسی»")

        if msg:
            self._remember(user_id, {'role': 'ai', 'text': msg})
        return {'reply': msg, 'intent': intent, 'is_edit_request': intent == 'edit'}

    def processing_line(self) -> str:
        return random.choice(PROCESSING_LINES)

    def success_line(self) -> str:
        return random.choice(SUCCESS_LINES)

    def no_face_line(self) -> str:
        return NO_FACE_TIPS[0]

    # ---------- کمکی ----------
    def _pick(self, uid: str, pool: List[str]) -> str:
        """انتخاب غیرتکراری نسبت به آخرین پاسخ همان دسته."""
        choice = random.choice(pool)
        for _ in range(6):
            if choice != self._last_replies.get(uid):
                break
            choice = random.choice(pool)
        self._last_replies[uid] = choice
        return choice

    def _remember(self, uid: str, item: dict) -> None:
        h = self._history.setdefault(uid, [])
        h.append(item)
        del h[:-24]                     # حافظه سبک: ۱۲ پیام آخر

    def history(self, uid: str) -> List[dict]:
        return list(self._history.get(uid, []))


CHAT_BOT = BeautyChatBot()
