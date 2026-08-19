# src/keywords_database.py
"""
دیتابیس کلیدواژه‌های عامیانه و حرفه‌ای برای BeautyAI
"""

# ============================================
# ۱. نواحی صورت
# ============================================
FACE_AREAS = {
    'nose': {'persian': ['بینی', 'دماغ', 'بینی‌ام', 'دماغم'], 'english': ['nose']},
    'lip': {'persian': ['لب', 'لبام', 'دهان'], 'english': ['lip']},
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
    'high': {'persian': ['زیاد', 'خیلی', 'بسیار'], 'english': ['high'], 'value': 0.8},
    'very_high': {'persian': ['خیلی زیاد', 'حداکثر'], 'english': ['very high'], 'value': 0.95}
}

# ============================================
# ۴. سلبریتی‌ها
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
        
        # ساخت کش
        self._build_cache()
    
    def _build_cache(self):
        """ساخت کش برای جستجوی سریع"""
        self.area_cache = {}
        self.action_cache = {}
        self.intensity_cache = {}
        self.celebrity_cache = {}
        
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
    
    def get_area(self, word: str) -> str:
        return self.area_cache.get(word.lower())
    
    def get_action(self, word: str) -> str:
        return self.action_cache.get(word.lower())
    
    def get_intensity(self, word: str) -> str:
        return self.intensity_cache.get(word.lower())
    
    def get_celebrity(self, word: str) -> str:
        return self.celebrity_cache.get(word.lower())
    
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