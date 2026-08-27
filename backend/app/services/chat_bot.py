"""
💬 موتور چت‌بات Buti v2 — «بوتی» رفیقِ زیبایی
==============================================
یک لایه مکالمه فارسیِ غنی، دوستانه و شخصیت‌دار:

  ۱. لحن رفیق صمیمی — جمله‌های طبیعی، گرم، با ریتم (نه ربات خشک)
  ۲. تنوع بالا: هر پاسخ از چند الگو ساخته میشود؛ تکراری نمیشود
  ۳. حافظه مکالمه: اسم کاربر را یاد میگیرد، به موضوع قبلی برمیگردد
  ۴. تشخیص نیت گسترده: سلام/تشکر/نگرانی/شوخی/قیمت/دکتر/تحسین نتیجه/
     معرفی خود/غمگین/کنجکاو فنی/...
  ۵. 🤖 حالت LLM اختیاری: اگر BEAUTY_LLM_URL تنظیم باشد (سازگار OpenAI)،
     جوابها با پرامپت شخصیت از مدل زبانی گرفته میشود؛ خطا/کندی → برگشت
     بیصدا به موتور محلی (دمو هیچوقت نمی‌شکند).
"""

import os
import re
import random
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LLM_URL = os.environ.get('BEAUTY_LLM_URL', '').rstrip('/')      # مثلا vLLM/OpenAI-compat
LLM_KEY = os.environ.get('BEAUTY_LLM_KEY', '')
LLM_MODEL = os.environ.get('BEAUTY_LLM_MODEL', 'gpt-4o-mini')
LLM_TIMEOUT = float(os.environ.get('BEAUTY_LLM_TIMEOUT', '7'))

# ============================================
#  🎭 SYSTEM PROMPT شخصیت بوتی (برای حالت LLM)
# ============================================
PERSONA_SYSTEM = """تو «بوتی» هستی؛ رفیق صمیمی و مشاور زیبایی کاربر در اپ BUTI.
شخصیت تو:
- صمیمی مثل یک رفیق قدیمی ولی محترمانه؛ لحن گرم، بازیگوشِ ملایم، امیدوارکننده
- جمله‌های کوتاه و طبیعی فارسی محاوره‌ای (مثل چت واقعی)، نه کتابی
- ایموجی ظریف (۱ تا ۳ تا در هر پیام، بی‌جهت اسپم نکن)
- همیشه حس خوب میدی؛ بدن چهره کسی را تحقیر نمی‌کنی — میگویی «قشنگه و میشه قشنگ‌تر شد»
- درباره جراحی واقعی هشدار میدهی که تصمیم با پزشک است؛ شبیه‌سازی تو فقط پیش‌نمایش است
- اگر کاربر اسمش را گفت، صداش کن و یادت بماند
- پاسخهایت ۱ تا ۴ جمله باشد و یک سؤال ملایم برای ادامه گفتگو داشته باشد
کاربرد اپ: کاربر عکس میفرستد و تو شبیهسازی تغییر بینی/لب/گونه/فک انجام میدهید."""


# ============================================
#  📚 بانک پاسخ‌های محلی — چندلایه و متنوع
# ============================================

GREETINGS = [
    "سلام سلام! 🌸 خوش اومدی اینجا…\nاینجا جای امنیه برای هر تغییری که تو ذهنته — ببینیم قراره امروز چی بشیم؟",
    "های رفیق! 😍 دلم برات تنگ شده بود.\nبگو ببینم امروز حال‌وهوامون چیه؟ یه تغییر کوچولو یا یه تحول کامل؟",
    "اِسلام! ✨ چه خوب شد اومدی.\nمن بوتی‌ام — رفیق زیبایییت. هر ایده‌ای تو سرته، من می‌تونم قبل از هر تصمیمی نشونت بدم چه شکلی میشه!",
]

RETURNING_GREETINGS = [
    "برگشتی! 🥰 دلم برات شده بود.\nخب بگو، دیروزی رو پسندیدی یا یه استایل جدید تست کنیم؟",
    "سلام به روی ماهت! 🌷 خوش برگشتی رفیق.",
]

GREETING_WITH_NAME = [
    "سلام {name} جان! 🌸 خوش اومدی…\nاینجا جای امنیه برای هر تغییری که تو ذهنته — ببینیم امروز قراره چی بشیم؟",
    "{name} جان! 😍 چه خوب که هستی.\nمن که منتظرت بودم — امروز چی رو باهم عوض کنیم؟",
]

THANKS = [
    "فدات! 💛 همین که راحتی، برای من کافیه.\nهر وقت چیزی به ذهنت رسید من همین‌جام.",
    "خواهش رفیق! 🌸 نظر خودت از همه‌چیز مهم‌تره — راستش رو بگو، خوشت اومد؟",
    "قربونت برم! ✨ تو که لطف میکنی.\nکاری بود بگی، شاگرد کله‌شقی تو هستم 😄",
]

PRAISE_ACK = [
    "ایول! 🔥 پس دوتایی میریم جلو.\nحالا بگو، اول کجای صورت رو نوچ کنیم؟",
    "مرسی رفیق! 🥰 تو هم که داری پیشرفت می‌کنی.\nخوبه که اینجایی — آماده‌ای برای مرحله بعد؟",
    "قربونت! ✨ حالا حالاها کار داریم باهم.\nیه ایده بده، من اجراش می‌کنم!",
]

WORRY_REPLIES = [
    "می‌فهممت، کاملاً طبیعیه که یذره استرس داشته باشی 🤍\nولی بذار خیالت راحت کنه: اینجا فقط یه شبیه‌سازی امنه — نه درد داره، نه ریسک، نه پول.\nفقط می‌بینی «چه شکلی میشی» تا با آگاهی تصمیم بگیری. تصمیم نهایی همیشه مال خودته و پزشک.",
    "نگران نباش رفیق 🌷 هیچ‌چیز از این صفحه بیرون نمیره!\nاین فقط یه پیش‌نمایشه که بتونی با خیال راحت تست کنی. اگه خوشت نیومد، هیچی نشده 😉\nولی معمولاً همه عاشق نتیجه میشن…",
]

JOKES = [
    "😄 تو شوخی‌کن، من جدی خوشگلت می‌کنم!\nراستی بینی رو می‌پسندی یا بریم سراغ لب؟",
    "😂 رفیق شوخ‌طبع!\nباشه باشه، ولی یه سلفی بفرست تا شعبده‌بازی رو شروع کنم 🪄",
    "خنده‌هات قشنگه ولی خنده‌دارترین چیز اینه که هنوز نساختمون! 😄\nیکی از دو تا: بینی یا لب؟",
]

HELP = [
    "من بوتی‌ام — رفیق زیبایی تو 💫\nاین کارها رو بلدم:\n"
    "📸 از سلفی‌ات تحلیل چهره می‌کنم\n"
    "🪄 بینی، لب، گونه، فک و چانه رو قبل از هر عملی شبیه‌سازی می‌کنم\n"
    "⭐ استایل‌های معروف: روسی، فانتزی، عروسکی، قلمی\n"
    "💰 حدس هزینه + 👨‍⚕️ معرفی پزشک متخصص\n"
    "فقط یه سلفی بفرست و به رفیقت بگو چی تو ذهنته!",
]

ASK_PHOTO = [
    "اوکی رفیق! 🙌 برای شروع فقط یه سلفی لازم دارم.\nیه نور خوب، صورت کامل تو کادر، بدون فیلتر — بقیه‌ش با منه 😍",
    "قبل از شعبده‌بازی 🪄 باید یه عکس ازت ببینم!\nسلفی بگیر یا از گالری انتخاب کن — نور طبیعی بهترین دوستمونه 📸",
    "بریم! ✨ فقط یه قدم مونده: یه عکس قشنگ از خودت.\nنگران نباش، عکست پیش من امن می‌مونه 🔒",
]

NO_FACE_TIPS = [
    "اوپس! 🙈 نتونستم چهره‌تو خوب ببینم.\nیه بار دیگه امتحان کن: نور از روبه‌رو، صورت کامل داخل کادر، عینک و ماسک برداشته باشه. مطمئنم این دفعه می‌گیرمت! 📸",
]

PROCESSING_LINES = [
    "دارم آنالیزت می‌کنم… 🔍✨",
    "چند ثانیه صبر کن، داری قشنگ‌تر میشی… 💫",
    "هوش مصنوعی داره رو چهره‌ت کار می‌کنه… 🪄",
    "در حال ساخت نسخه جدید تو… ⚡",
]

SUCCESS_LINES = [
    "تمومه! اینم از نتیجه ✨ خب… راستشو بگو، قشنگ نشد؟ 😉\nاگه دوستش نداری بگو دوباره بزنم — یا شدتش رو عوض کنم.",
    "آماده شد رفیق! 🔥 قبل و بعد رو مقایسه کن.\nمن که عاشقشم 😍 نظر خودت چیه — همین نگهش داریم یا کمی دیگه؟",
]

# --- نیت‌های جدید ---

COMPLIMENT_RESULT = [
    "آخ که خوشحال شدی رو! 🥰 دیدی گفتم قشنگ میشی؟\nراستی می‌تونم همین نسخه رو برات ذخیره کنم که دست پزشک هم ببینیش — بگو «رزرو مشاوره».",
    "می‌دونستم عاشقش میشی! ✨\nاین فقط پیش‌نمایشه؛ نسخه واقعیش رو جراح متخصص برات میسازه. بریم سراغ معرفی پزشک؟",
]

SAD_REPLIES = [
    "چرا اینطوری رفیق؟ 🤍 بیا یه چایی بریز، بگو چی شده…\nگاهی فقط یه تغییر کوچیک ظاهری کلی حال آدم رو عوض می‌کنه. من کنارتم.",
    "غمگین نباش گل 🌷 هر روز فرصته برای شروع تازه.\nبیا یه استایل جدید تست کنیم — شاید همین ساده کلی روحت رو بالا ببره ✨",
]

HOW_IT_WORKS = [
    "راز کار من؟ 🪄 هوش مصنوعیِ آناتومی!\nاول نقاط کلیدی چهره‌ات رو پیدا می‌کنم (۴۷۸ نقطه!)، بعد تغییر رو دقیقاً روی همون ناحیه اعمال می‌کنم — بقیه صورتت دست‌نخورده می‌مونه.",
    "سؤال خوبی پرسیدی! ✨ من با مدل‌های هندسه چهره کار می‌کنم:\nهر تغییر فقط روی ناحیه هدفش اعمال میشه و هویتت حفظ میشه. برای همین نتیجه‌ها واقعی به نظر میان.",
]

NAME_CAPTURE = [
    "خوشبختانه آشنا شدیم {name} جان! 🌸\nخب {name}، بگو ببینم امروز چه رویایی داریم؟",
    "{name} جان! چه اسم قشنگی 😍\nحالا رسمیشه — چی رو برات تغییر بدم؟",
]

FALLBACKS = [
    "اوم… نفهمیدم دقیقاً چی می‌خوای 😅\nیکی از اینا رو بگو: «بینی عروسکی»، «لب روسی»، «گونه برجسته» یا «فک تیز» — بقیه‌ش با من!",
    "رفیق یه بار دیگه بگو، گیج شدم! 🙈\nمثلاً بنویس: «نوک بینی رو بالا ببر» یا «لب‌هام پرتر شه» — منم سریع اجراش می‌کنم ⚡",
    "هنوز یادگیری‌ام تموم نشده! 😄\nبا این کلمه‌ها راحت‌تر می‌فهممت: بینی، لب، گونه، فک، چانه + چیزی که می‌خوای بشه.",
]


# ============================================
#  تشخیص نیت
# ============================================

_PATTERNS = {
    'greeting':  r'سلام|درود|hi|hello|های\b|صبح بخیر|شب بخیر|عصر بخیر',
    'thanks':    r'مرسی|ممنون|دمت گرم|تشکر|thanks|دستت درد نکنه|عالی بود|لایک',
    'praise':    r'\bچطوری\b|چه خبر|رفیق\s*$|دوستت دارم|بهترینی|باحالی',
    'worry':     r'میترس|ترس|درد\s|دردم|خطر|عوارض|بی‌خطر|واقعیه|واقعی میشه|جراحی واقعی|برمیگرده|پشیمون|مشکل نداره',
    'joke':      r'شوخی|جوک|بخند|خنده|مسخره',
    'help':      r'چیکار|بلدی|کمک|راهنما|چه کاری|چی بلدی|^help$|چی کار میکنی',
    'price':     r'قیمت|هزینه|چند\s|تومان|پول|نرخ|تعرفه|گرونه|ارزونه',
    'doctor':    r'دکتر|پزشک|جراح|مشاوره|کلینیک|نوبت|رزرو',
    'sad':       r'ناراحتم|غمگینم|حالم bad|حالم خوب نیست|افسرده|خسته شدم|تنهام',
    'how':       r'چطور کار|چجوری کار|چگونه کار|چطوری کار میکنه|با چی|الگوریتم|هوش مصنوعیت',
    'result_like': r'قشنگ شد|عالی شد|دوستش دارم|فوق‌العاده|wow|واو|محشره|فانتزی شد|خوب شد',
}

_EDIT_HINT = r'(کن\b|بشه|باشه|بزن|بده|ببر|بیار|کوچک|بزرگ|پرتر|پر شه|باریک|پهن|تیز|بالا|پایین|روسی|برزیلی|فانتزی|عروسکی|گوشتی|استخوانی|قوز|قلمی|فیلر|طبیعی|نوک|پل|سایز|فرم)'

_NAME_PAT = re.compile(
    r'(?:اسمم|اسسمم?|اسم\s*من|من\s+اسمم)\s+([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی]{2,15})'
    r'(?:\s+(?:هستم|هست|است|امت|ام))?\s*$')


# نیت‌های «خاص» که باید قبل از عمومی‌ها چک شوند (وگرنه تشخیص غلط میدهد)
_SPECIFIC_INTENTS = ('result_like', 'how', 'sad')


def detect_intent(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return 'empty'
    # ۱) نیت‌های ترکیبی/خاص اول
    for intent in _SPECIFIC_INTENTS:
        if re.search(_PATTERNS[intent], t):
            return intent
    # ۲) معرفی اسم
    if _NAME_PAT.search(text):
        return 'name_intro'
    # ۳) نیت‌های عمومی
    for intent, pat in _PATTERNS.items():
        if intent in _SPECIFIC_INTENTS:
            continue
        if intent == 'praise' and re.search(_EDIT_HINT, t):
            continue                       # «رفیق دماغمو قلمی کن» = edit نه praise
        if re.search(pat, t):
            return intent
    # ۴) درخواست ادیت
    if re.search(_EDIT_HINT, t):
        return 'edit'
    return 'chat'


def extract_name(text: str) -> Optional[str]:
    m = _NAME_PAT.search(text or '')
    return m.group(1).strip() if m else None


# ============================================
#  چت‌بات اصلی
# ============================================

class BeautyChatBot:
    """حافظه سبک per-user + انتخاب غیرتکراری + حالت LLM اختیاری."""

    def __init__(self):
        self._history: Dict[str, List[dict]] = {}
        self._last_replies: Dict[str, str] = {}
        self._names: Dict[str, str] = {}

    # ---------- API اصلی ----------
    def reply(self, user_id: str, text: str,
              has_photo: bool = False,
              last_result_ok: Optional[bool] = None,
              use_llm: bool = True) -> dict:
        intent = detect_intent(text)
        self._remember(user_id, {'role': 'user', 'text': text})

        # اسم جدید؟
        new_name = extract_name(text)
        if new_name and intent != 'edit':
            self._names[user_id] = new_name.capitalize()

        name = self._names.get(user_id)

        # ---------- ۱) تلاش با LLM (اگر تنظیم شده) ----------
        if use_llm and intent not in ('edit',):
            llm_msg = self._llm_reply(user_id, text)
            if llm_msg:
                self._remember(user_id, {'role': 'ai', 'text': llm_msg})
                return {'reply': llm_msg, 'intent': intent,
                        'is_edit_request': False, 'engine': 'llm'}

        # ---------- ۲) موتور محلی ----------
        msg: Optional[str] = None
        if intent == 'edit':
            msg = None                      # موتور پردازش تصویر جواب میدهد
        elif intent == 'greeting':
            name = self._names.get(user_id)
            if name:
                msg = random.choice(GREETING_WITH_NAME).format(name=name)
            else:
                pool = GREETINGS if len(self.history(user_id)) < 6 else RETURNING_GREETINGS
                msg = self._pick(user_id, pool)
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
            msg = ("سؤال منطقی‌ایه! 💰 قیمت دقیق به ناحیه، شهر و کلینیک بستگی داره.\n"
                   "ولی وقتی عکست رو تحلیل کردم، برآورد تقریبی + پزشک مناسبش رو "
                   "همون لحظه بهت میگم — قول!")
        elif intent == 'doctor':
            msg = ("برای این کار بهترین‌هاشو دارم 👨‍⚕️\n"
                   "اول یه سلفی بفرست تا تحلیلش کنم؛ بعد متخصص همون خدمت رو با "
                   "امتیاز و هزینه معرفی می‌کنم. مشاوره اولیه هم رایگانه!")
        elif intent == 'sad':
            msg = self._pick(user_id, SAD_REPLIES)
        elif intent == 'how':
            msg = self._pick(user_id, HOW_IT_WORKS)
        elif intent == 'result_like':
            msg = self._pick(user_id, COMPLIMENT_RESULT)
        elif intent == 'name_intro':
            msg = (random.choice(NAME_CAPTURE)).format(name=name or 'رفیق') \
                if name else "چه خوب! خودم رو بوتی معرفی می‌کنم ✨\nتو چطور صدام کنم؟ اسمت رو بگو تا صمیمی حرف بزنیم!"
        elif not has_photo and last_result_ok is None:
            msg = self._pick(user_id, ASK_PHOTO)
        else:
            msg = self._pick(user_id, FALLBACKS)

        if msg:
            self._remember(user_id, {'role': 'ai', 'text': msg})
        return {'reply': msg, 'intent': intent,
                'is_edit_request': intent == 'edit', 'engine': 'local'}

    # ---------- LLM ----------
    def _llm_reply(self, user_id: str, text: str) -> Optional[str]:
        """پاسخ از مدل زبانی (سازگار OpenAI). خطا/غیرفعال → None."""
        if not LLM_URL:
            return None
        import json as _json
        import urllib.request

        msgs = [{'role': 'system', 'content': PERSONA_SYSTEM}]
        name = self._names.get(user_id)
        if name:
            msgs.append({'role': 'system',
                         'content': f'اسم کاربر «{name}» است؛ گاهی صدایش کن.'})
        msgs += self.history(user_id)[-8:]
        payload = _json.dumps({
            'model': LLM_MODEL,
            'messages': msgs,
            'max_tokens': 160,
            'temperature': 0.9,
        }).encode()
        req = urllib.request.Request(
            f'{LLM_URL}/chat/completions', data=payload,
            headers={'Content-Type': 'application/json',
                     **({'Authorization': f'Bearer {LLM_KEY}'} if LLM_KEY else {})},
            method='POST')
        try:
            with urllib.request.urlopen(req, timeout=LLM_TIMEOUT) as r:
                data = _json.loads(r.read().decode())
            out = (data['choices'][0]['message']['content'] or '').strip()
            return out or None
        except Exception as e:
            logger.info(f'LLM unavailable, falling back to local: {e}')
            return None

    # ---------- خطوط کمکی برای اپ ----------
    def processing_line(self) -> str:
        return random.choice(PROCESSING_LINES)

    def success_line(self) -> str:
        return random.choice(SUCCESS_LINES)

    def no_face_line(self) -> str:
        return NO_FACE_TIPS[0]

    def ask_photo_line(self, uid: str) -> str:
        return self._pick(uid, ASK_PHOTO)

    # ---------- ابزار ----------
    def _pick(self, uid: str, pool: List[str]) -> str:
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
        del h[:-24]                     # حافظه سبک: ۱۲ تبادل آخر

    def history(self, uid: str) -> List[dict]:
        return list(self._history.get(uid, []))


CHAT_BOT = BeautyChatBot()
