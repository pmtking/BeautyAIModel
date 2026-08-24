# src/keywords_database.py
"""
دیتابیس کلیدواژه‌های عامیانه و حرفه‌ای برای BeautyAI
شامل انواع بینی (۱۵ استایل) و انواع لب (۱۰ استایل)
"""

# ============================================
# ۱. نواحی صورت
# ============================================
FACE_AREAS = {
    'nose': {'persian': ['بینی', 'دماغ', 'بینی‌ام', 'دماغم', 'منخر'], 'english': ['nose']},
    'lip': {'persian': ['لب', 'لبام', 'دهان', 'لبامو'], 'english': ['lip']},
    'jaw': {'persian': ['فک', 'چانه', 'فکم'], 'english': ['jaw']},
    'cheek': {'persian': ['گونه', 'گونم'], 'english': ['cheek']},
    'forehead': {'persian': ['پیشانی', 'پیشونم'], 'english': ['forehead']},
    'eye': {'persian': ['چشم', 'چشمام'], 'english': ['eye']}
}

# ============================================
# ۲. اعمال
# ============================================
ACTIONS = {
    'smaller': {'persian': ['کوچک‌تر', 'باریک‌تر', 'کم‌تر'], 'english': ['smaller']},
    'bigger': {'persian': ['بزرگ‌تر', 'بلندتر', 'بیشتر'], 'english': ['bigger']},
    'fuller': {'persian': ['پرتر', 'حجم‌تر'], 'english': ['fuller']},
    'sharper': {'persian': ['تیزتر', 'مشخص‌تر'], 'english': ['sharper']},
    'smoother': {'persian': ['صاف‌تر', 'یکدست'], 'english': ['smoother']},
    'lift': {'persian': ['لیفت', 'بالا'], 'english': ['lift']}
}

# ============================================
# ۳. شدت‌ها
# ============================================
INTENSITIES = {
    'very_low': {'persian': ['خیلی کم', 'کمترین'], 'english': ['very low'], 'value': 0.2},
    'low': {'persian': ['کم', 'یکم', 'ملایم'], 'english': ['low'], 'value': 0.4},
    'medium': {'persian': ['متوسط', 'معمولی'], 'english': ['medium'], 'value': 0.6},
    'high': {'persian': ['زیاد', 'خیلی'], 'english': ['high'], 'value': 0.8},
    'very_high': {'persian': ['خیلی زیاد', 'حداکثر'], 'english': ['very high'], 'value': 0.95}
}

# ============================================
# ۴. انواع بینی — ۱۵ استایل
#    id → {label فارسی, keywords فارسی/انگلیسی, توضیح}
# ============================================
NOSE_STYLES = {
    'fleshy': {
        'label': 'گوشتی',
        'keywords': ['گوشتی', 'پرپهن', 'مگسکی'],
        'description': 'نوک و بال‌های بینی پهن‌تر و پرتر',
    },
    'fantasy': {
        'label': 'فانتزی',
        'keywords': ['فانتزی', 'اروپایی'],
        'description': 'باریک شدید + نوک خیلی بالا (فرم اروپایی)',
    },
    'natural': {
        'label': 'طبیعی',
        'keywords': ['طبیعی', 'عادی'],
        'description': 'کمی باریک‌تر بدون تغییر محسوس نوک',
    },
    'bony': {
        'label': 'استخوانی',
        'keywords': ['استخوانی', 'تیزی', 'استخوانی تیز'],
        'description': 'پل تیز و باریک با ساختار زاویه‌دار',
    },
    'half_fantasy': {
        'label': 'نیمه‌فانتزی',
        'keywords': ['نیمه فانتزی', 'نیمه‌فانتزی', 'نصف فانتزی'],
        'description': 'بین طبیعی و فانتزی',
    },
    'doll_tip': {
        'label': 'عروسکی',
        'keywords': ['عروسکی', 'مزدانه', 'مزدانة', 'دخترانه عروسکی'],
        'description': 'نوک گرد، کوچک و کاملاً سربالا',
    },
    'upturned_tip': {
        'label': 'نوک بالا',
        'keywords': ['نوک بالا', 'نوکش بالا', 'سر بالا', 'نوک سربالا', 'بالای نوک'],
        'description': 'فقط چرخش نوک بینی به سمت بالا',
    },
    'filler': {
        'label': 'فیلر زده شده',
        'keywords': ['فیلر', 'فیلر زده', 'ژل بینی', 'تزریق بینی'],
        'description': 'پر شدن گودی رادیکس و پل بینی مثل بعد از فیلر',
    },
    'slim_bridge': {
        'label': 'قلمی',
        'keywords': ['قلمی', 'قلم مانند'],
        'description': 'پل بسیار باریک و ظریف + نوک کمی بالا',
    },
    'smaller': {
        'label': 'کوچک‌تر',
        'keywords': [],
        'description': 'باریک‌تر و کوتاه‌تر به‌طور یکجا',
    },
    'narrower': {
        'label': 'باریک‌تر',
        'keywords': [],
        'description': 'فقط کاهش پهنای بینی',
    },
    'shorter': {
        'label': 'کوتاه‌تر',
        'keywords': [],
        'description': 'فقط کوتاه کردن طول بینی',
    },
    'longer': {
        'label': 'بلندتر',
        'keywords': [],
        'description': 'افزایش طول بینی',
    },
    'wider': {
        'label': 'پهن‌تر',
        'keywords': [],
        'description': 'افزایش پهنای بال‌های بینی',
    },
    'bigger': {
        'label': 'بزرگ‌تر',
        'keywords': [],
        'description': 'بزرگ شدن کلی بینی',
    },
}

# ============================================
# ۵. انواع لب — ۱۰ استایل
# ============================================
LIP_STYLES = {
    'russian': {
        'label': 'روسی',
        'keywords': ['روسی', 'روس'],
        'description': 'حجم بالا + کمان کوپید مشخص — لب روسی معروف',
    },
    'brazilian': {
        'label': 'برزیلی',
        'keywords': ['برزیلی', 'برزیل'],
        'description': 'حجم بالا + گوشه‌های رو به بالا',
    },
    'hollywood': {
        'label': 'هالیوودی',
        'keywords': ['هالیوودی', 'هالیود', 'هالیوود'],
        'description': 'حجم زیاد + برجستگی مرکز لب پایین',
    },
    'heart_shape': {
        'label': 'قلوه‌ای',
        'keywords': ['قلوه‌ای', 'قلب'],
        'description': 'فرم قلب با کمان کوپید عمیق',
    },
    'classic': {
        'label': 'کلاسیک',
        'keywords': ['کلاسیک'],
        'description': 'حجم متعادل و فرم کلاسیک',
    },
    'natural': {
        'label': 'طبیعی',
        'keywords': ['طبیعی'],
        'description': 'حجم ملایم روزمره',
    },
    'fuller': {
        'label': 'پرتر',
        'keywords': [],
        'description': 'افزایش حجم یکنواخت هر دو لب',
    },
    'thinner': {
        'label': 'باریک‌تر',
        'keywords': [],
        'description': 'کاهش حجم لب',
    },
    'cupids_bow': {
        'label': 'کمان کوپید تیز',
        'keywords': ['کمان کوپید', 'کوپید'],
        'description': 'تعریف کمان کوپید بدون حجم زیاد',
    },
    'corner_lift': {
        'label': 'گوشه‌ها بالا',
        'keywords': ['گوشه لب بالا', 'لبخند لب'],
        'description': 'لیفت گوشه‌های لب — لبخند بدون جراحی',
    },
}

# ============================================
# ۶. سلبریتی‌ها
# ============================================
CELEBRITIES = {
    'angelina_jolie': {
        'name': 'آنجلینا جولی',
        'features': {'lip': 'لب‌های پر', 'cheek': 'گونه‌های برجسته'},
        'keywords': ['آنجلینا', 'جولی']
    },
    'brad_pitt': {
        'name': 'برد پیت',
        'features': {'jaw': 'فک تیز', 'nose': 'بینی صاف'},
        'keywords': ['برد', 'پیت']
    },
    'scarlett_johansson': {
        'name': 'اسکارلت جوهانسون',
        'features': {'lip': 'لب‌های پر', 'eye': 'چشم‌های درشت'},
        'keywords': ['اسکارلت', 'جوهانسون']
    }
}


class KeywordDatabase:
    """مدیریت دیتابیس کلیدواژه‌ها"""

    def __init__(self):
        self.areas = FACE_AREAS
        self.actions = ACTIONS
        self.intensities = INTENSITIES
        self.celebrities = CELEBRITIES
        self.nose_styles = NOSE_STYLES
        self.lip_styles = LIP_STYLES

        # ساخت کش
        self._build_cache()

    def _build_cache(self):
        """ساخت کش برای جستجوی سریع"""
        self.area_cache = {}
        self.action_cache = {}
        self.intensity_cache = {}
        self.celebrity_cache = {}
        self.nose_style_cache = {}
        self.lip_style_cache = {}

        for key, value in self.areas.items():
            for word in value.get('persian', []) + value.get('english', []):
                self.area_cache[word] = key

        for key, value in self.actions.items():
            for word in value.get('persian', []) + value.get('english', []):
                self.action_cache[word] = key

        for key, value in self.intensities.items():
            for word in value.get('persian', []) + value.get('english', []):
                self.intensity_cache[word] = key

        for key, value in self.celebrities.items():
            for word in value.get('keywords', []):
                self.celebrity_cache[word.lower()] = key

        for key, value in self.nose_styles.items():
            for word in value.get('keywords', []):
                self.nose_style_cache[word] = key

        for key, value in self.lip_styles.items():
            for word in value.get('keywords', []):
                self.lip_style_cache[word] = key

    def get_area(self, word: str) -> str:
        return self.area_cache.get(word.lower())

    def get_action(self, word: str) -> str:
        return self.action_cache.get(word.lower())

    def get_intensity(self, word: str) -> str:
        return self.intensity_cache.get(word.lower())

    def get_celebrity(self, word: str) -> str:
        return self.celebrity_cache.get(word.lower())

    def get_nose_style(self, text: str):
        """جستجوی استایل بینی در متن — طولانی‌ترین تطابق."""
        best, best_len = None, 0
        for kw, style in self.nose_style_cache.items():
            if kw in text and len(kw) > best_len:
                best, best_len = style, len(kw)
        return best

    def get_lip_style(self, text: str):
        """جستجوی استایل لب در متن — طولانی‌ترین تطابق."""
        best, best_len = None, 0
        for kw, style in self.lip_style_cache.items():
            if kw in text and len(kw) > best_len:
                best, best_len = style, len(kw)
        return best

    def get_intensity_value(self, level: str) -> float:
        return self.intensities.get(level, {}).get('value', 0.5)

    def get_all_areas(self) -> list:
        return list(self.areas.keys())

    def get_all_actions(self) -> list:
        return list(self.actions.keys())


# نمونه سراسری
keyword_db = KeywordDatabase()

if __name__ == "__main__":
    print("✅ KeywordDatabase initialized")
    print(f"Areas: {len(keyword_db.areas)}")
    print(f"Actions: {len(keyword_db.actions)}")
    print(f"Nose styles: {len(keyword_db.nose_styles)}")
    print(f"Lip styles: {len(keyword_db.lip_styles)}")
