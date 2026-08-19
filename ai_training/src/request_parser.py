# src/request_parser.py
"""
پارسر هوشمند درخواست‌های کاربر
"""

from src.keywords_database import keyword_db


class RequestParser:
    """پارسر درخواست‌های کاربر"""
    
    def __init__(self):
        self.db = keyword_db
        self.areas = self.db.get_all_areas()
        self.actions = self.db.get_all_actions()
    
    def parse(self, text: str) -> dict:
        """تحلیل درخواست کاربر"""
        text_lower = text.lower()
        words = text_lower.split()
        
        result = {
            'area': None,
            'action': None,
            'intensity': 0.5,
            'intensity_level': 'medium',
            'celebrity': None,
            'is_routine': False,
            'is_consultation': False,
            'message': '',
            'confidence': 0.0
        }
        
        # 1. تشخیص ناحیه
        for word in words:
            area = self.db.get_area(word)
            if area:
                result['area'] = area
                break
        
        # 2. تشخیص عمل
        for word in words:
            action = self.db.get_action(word)
            if action:
                result['action'] = action
                break
        
        # 3. تشخیص شدت
        for word in words:
            intensity = self.db.get_intensity(word)
            if intensity:
                result['intensity'] = self.db.get_intensity_value(intensity)
                result['intensity_level'] = intensity
                break
        
        # 4. تشخیص روتین پوستی
        if 'روتین' in text_lower or 'مراقبت' in text_lower:
            result['is_routine'] = True
        
        # 5. تشخیص مشاوره
        if 'مشاوره' in text_lower or 'نظر' in text_lower:
            result['is_consultation'] = True
        
        # 6. تشخیص سلبریتی
        for word in words:
            celeb = self.db.get_celebrity(word)
            if celeb:
                celeb_data = self.db.celebrities.get(celeb, {})
                result['celebrity'] = celeb_data.get('name')
                break
        
        # 7. تولید پیام
        result['message'] = self._generate_message(result)
        
        # 8. محاسبه اطمینان
        result['confidence'] = self._calculate_confidence(result)
        
        return result
    
    def _generate_message(self, result: dict) -> str:
        """تولید پیام توضیحی"""
        if result['area'] and result['action']:
            area_names = {
                'nose': 'بینی', 'lip': 'لب', 'jaw': 'فک',
                'cheek': 'گونه', 'forehead': 'پیشانی', 'eye': 'چشم'
            }
            action_names = {
                'smaller': 'کوچک‌تر', 'bigger': 'بزرگ‌تر', 'fuller': 'پرتر',
                'sharper': 'تیزتر', 'smoother': 'صاف‌تر', 'lift': 'لیفت'
            }
            area_name = area_names.get(result['area'], result['area'])
            action_name = action_names.get(result['action'], result['action'])
            intensity_percent = int(result['intensity'] * 100)
            return f"✅ {area_name} با شدت {intensity_percent}% {action_name} می‌شود!"
        
        if result['is_routine']:
            return "🧴 روتین پوستی برای شما آماده شد!"
        
        if result['is_consultation']:
            return "👨‍⚕️ مشاوره زیبایی برای شما آماده شد!"
        
        if result['celebrity']:
            return f"🌟 الهام از {result['celebrity']}!"
        
        return "❌ متوجه نشدم! لطفاً دقیق‌تر بگویید."
    
    def _calculate_confidence(self, result: dict) -> float:
        """محاسبه اطمینان"""
        score = 0
        total = 0
        
        if result['area']:
            score += 1
        total += 1
        
        if result['action']:
            score += 1
        total += 1
        
        if result['celebrity']:
            score += 0.5
        total += 0.5
        
        return round(score / total, 2) if total > 0 else 0.0


# ✅ ایجاد نمونه سراسری
parser = RequestParser()


# تست
if __name__ == "__main__":
    print("🧪 Testing RequestParser...")
    
    test_texts = [
        "دماغم رو کمی کوچیکتر کن",
        "لبامو پرتر کن",
        "روتین پوستی میخوام",
        "مشاوره زیبایی"
    ]
    
    for text in test_texts:
        result = parser.parse(text)
        print(f"\n📝 '{text}'")
        print(f"   Area: {result['area']}")
        print(f"   Action: {result['action']}")
        print(f"   Intensity: {result['intensity']}")
        print(f"   Message: {result['message']}")