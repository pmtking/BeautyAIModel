# src/models/consultation/service.py
"""
سیستم مشاوره زیبایی و روتین پوستی
"""

import random
from typing import Dict, List, Optional
from logger import logger


class ConsultationService:
    """سرویس مشاوره زیبایی"""
    
    def __init__(self):
        # انواع پوست
        self.skin_types = {
            'normal': 'معمولی',
            'dry': 'خشک',
            'oily': 'چرب',
            'combination': 'مختلط',
            'sensitive': 'حساس'
        }
        
        # مشکلات پوستی
        self.skin_issues = {
            'acne': 'جوش و آکنه',
            'pigmentation': 'لک و تیرگی',
            'wrinkle': 'چروک و خطوط',
            'dryness': 'خشکی و پوسته‌پوسته شدن',
            'oiliness': 'چربی و براقیت',
            'redness': 'قرمزی و حساسیت',
            'blackhead': 'جوش‌های سرسیاه',
            'dark_circle': 'گودی و تیرگی زیر چشم'
        }
        
        # محصولات پیشنهادی
        self.products = {
            'cleanser': {
                'name': 'پاک‌کننده',
                'types': {
                    'dry': 'کرم پاک‌کننده مرطوب‌کننده',
                    'oily': 'ژل پاک‌کننده ضد چربی',
                    'sensitive': 'پاک‌کننده ملایم بدون عطر',
                    'combination': 'پاک‌کننده متعادل‌کننده'
                }
            },
            'toner': {
                'name': 'تونر',
                'types': {
                    'dry': 'تونر مرطوب‌کننده',
                    'oily': 'تونر قابض منافذ',
                    'sensitive': 'تونر کالامین',
                    'combination': 'تونر متعادل‌کننده'
                }
            },
            'serum': {
                'name': 'سرم',
                'types': {
                    'acne': 'سرم ضد جوش با سالیسیلیک اسید',
                    'pigmentation': 'سرم ویتامین C',
                    'wrinkle': 'سرم رتینول',
                    'dryness': 'سرم هیالورونیک اسید',
                    'oiliness': 'سرم نیاسینامید'
                }
            },
            'moisturizer': {
                'name': 'مرطوب‌کننده',
                'types': {
                    'dry': 'کرم چرب و مغذی',
                    'oily': 'ژل مرطوب‌کننده سبک',
                    'sensitive': 'کرم بدون عطر و پارابن',
                    'combination': 'کرم متعادل‌کننده'
                }
            },
            'sunscreen': {
                'name': 'ضدآفتاب',
                'types': {
                    'normal': 'SPF 50+ ضدآفتاب روزانه',
                    'dry': 'ضدآفتاب مرطوب‌کننده SPF 50+',
                    'oily': 'ضدآفتاب مات‌کننده SPF 50+',
                    'sensitive': 'ضدآفتاب فیزیکی SPF 50+'
                }
            }
        }
        
        # برندهای معتبر
        self.brands = [
            'لاروش پوزای', 'سراوی', 'سینره', 'اوریجینز', 'کلینیک',
            'بیودرما', 'ویشی', 'ای‌آر‌دی', 'سلین', 'دکتر نوترینا'
        ]
        
        logger.info("✅ ConsultationService initialized")
    
    def analyze_skin(self, 
                     skin_type: Optional[str] = None,
                     issues: Optional[List[str]] = None,
                     age: Optional[int] = None,
                     gender: Optional[str] = None) -> Dict:
        """
        تحلیل وضعیت پوست و تولید توصیه‌ها
        
        Args:
            skin_type: نوع پوست
            issues: مشکلات پوستی
            age: سن
            gender: جنسیت
        
        Returns:
            dict: توصیه‌های کامل
        """
        # تنظیم پیش‌فرض‌ها
        if skin_type is None:
            skin_type = random.choice(list(self.skin_types.keys()))
        
        if issues is None:
            issues = random.sample(list(self.skin_issues.keys()), 2)
        
        result = {
            'skin_type': {
                'key': skin_type,
                'name': self.skin_types.get(skin_type, 'نامشخص')
            },
            'issues': [
                {'key': issue, 'name': self.skin_issues.get(issue, issue)}
                for issue in issues
            ],
            'recommendations': {
                'routine': self._generate_routine(skin_type, issues),
                'products': self._recommend_products(skin_type, issues),
                'tips': self._generate_tips(skin_type, issues, age, gender),
                'ingredients': self._recommend_ingredients(issues)
            },
            'warning': self._get_warnings(issues)
        }
        
        return result
    
    def _generate_routine(self, skin_type: str, issues: List[str]) -> Dict:
        """تولید روتین پوستی"""
        return {
            'morning': [
                '🧼 شستشو با پاک‌کننده مناسب',
                '💧 استفاده از تونر',
                f'🧪 سرم {self._get_best_serum(issues)}',
                '🧴 مرطوب‌کننده',
                '☀️ ضدآفتاب SPF 50+'
            ],
            'evening': [
                '🧼 پاک‌کننده دو مرحله‌ای',
                '💧 تونر',
                f'🧪 سرم شب {self._get_night_serum(issues)}',
                '🧴 مرطوب‌کننده شب',
                '👁️ کرم دور چشم'
            ],
            'weekly': [
                '🧖 هفته‌ای ۱-۲ بار ماسک صورت',
                '🧹 هفته‌ای ۱ بار اسکراب ملایم',
                '💆 هفته‌ای ۱ بار ماساژ صورت'
            ]
        }
    
    def _get_best_serum(self, issues: List[str]) -> str:
        """انتخاب بهترین سرم بر اساس مشکلات"""
        if 'acne' in issues:
            return 'سرم سالیسیلیک اسید'
        elif 'pigmentation' in issues:
            return 'سرم ویتامین C'
        elif 'wrinkle' in issues:
            return 'سرم رتینول'
        elif 'dryness' in issues:
            return 'سرم هیالورونیک اسید'
        else:
            return 'سرم نیاسینامید'
    
    def _get_night_serum(self, issues: List[str]) -> str:
        """انتخاب سرم شب"""
        if 'wrinkle' in issues:
            return 'رتینول ۰.۵%'
        elif 'pigmentation' in issues:
            return 'سرم روشن‌کننده با آربوتین'
        else:
            return 'سرم ترمیم‌کننده'
    
    def _recommend_products(self, skin_type: str, issues: List[str]) -> List[Dict]:
        """توصیه محصولات"""
        recommendations = []
        
        for product_key, product_data in self.products.items():
            product_type = product_data['types'].get(skin_type)
            if product_type is None:
                product_type = product_data['types'].get('normal', product_data['types'].get('combination'))
            
            # انتخاب برند تصادفی
            brand = random.choice(self.brands)
            
            recommendations.append({
                'category': product_data['name'],
                'product': product_type,
                'brand': brand,
                'reason': f'مناسب برای {self.skin_types.get(skin_type, "")}'
            })
        
        return recommendations
    
    def _generate_tips(self, skin_type: str, issues: List[str], 
                        age: Optional[int], gender: Optional[str]) -> List[str]:
        """تولید نکات مراقبتی"""
        tips = [
            '💧 روزانه ۸ لیوان آب بنوشید',
            '🥗 مصرف میوه و سبزیجات تازه را افزایش دهید',
            '😴 حداقل ۷-۸ ساعت خواب کافی داشته باشید',
            '🏃 ورزش منظم داشته باشید',
            '🚭 از استرس و سیگار دوری کنید'
        ]
        
        # نکات اختصاصی بر اساس مشکلات
        issue_tips = {
            'acne': [
                '🔹 از لمس کردن صورت خودداری کنید',
                '🔹 روبالشی خود را هر هفته عوض کنید',
                '🔹 از محصولات غیرکومدوژنیک استفاده کنید'
            ],
            'pigmentation': [
                '🔹 همیشه از ضدآفتاب استفاده کنید',
                '🔹 از محصولات روشن‌کننده استفاده کنید',
                '🔹 از لایه‌برداری منظم غافل نشوید'
            ],
            'wrinkle': [
                '🔹 از کرم‌های ضدچروک استفاده کنید',
                '🔹 ماساژ صورت را فراموش نکنید',
                '🔹 از مصرف زیاد قند خودداری کنید'
            ],
            'dryness': [
                '🔹 از مرطوب‌کننده‌های قوی استفاده کنید',
                '🔹 از شوینده‌های ملایم استفاده کنید',
                '🔹 در محیط از دستگاه بخور استفاده کنید'
            ]
        }
        
        for issue in issues:
            if issue in issue_tips:
                tips.extend(issue_tips[issue])
        
        # نکات بر اساس سن
        if age is not None:
            if age < 25:
                tips.append('🔸 تمرکز بر آبرسانی و پیشگیری')
            elif 25 <= age < 35:
                tips.append('🔸 شروع استفاده از ضدچروک')
            elif 35 <= age < 45:
                tips.append('🔸 استفاده از محصولات ضد پیری قوی‌تر')
            else:
                tips.append('🔸 استفاده از محصولات ترمیم‌کننده و مغذی')
        
        return tips
    
    def _recommend_ingredients(self, issues: List[str]) -> List[Dict]:
        """توصیه مواد مؤثر"""
        ingredients = {
            'acne': [
                {'name': 'سالیسیلیک اسید', 'benefit': 'لایه‌برداری و ضد جوش'},
                {'name': 'نیاسینامید', 'benefit': 'کاهش التهاب و چربی'},
                {'name': 'روغن درخت چای', 'benefit': 'ضد باکتری و ضد جوش'}
            ],
            'pigmentation': [
                {'name': 'ویتامین C', 'benefit': 'روشن‌کننده و آنتی‌اکسیدان'},
                {'name': 'آربوتین', 'benefit': 'کاهش لک‌ها'},
                {'name': 'کوجیک اسید', 'benefit': 'روشن‌کننده طبیعی'}
            ],
            'wrinkle': [
                {'name': 'رتینول', 'benefit': 'کاهش چروک و جوان‌سازی'},
                {'name': 'پپتیدها', 'benefit': 'تقویت کلاژن‌سازی'},
                {'name': 'هیالورونیک اسید', 'benefit': 'آبرسانی عمیق و پرشدن چروک'}
            ],
            'dryness': [
                {'name': 'هیالورونیک اسید', 'benefit': 'آبرسانی عمیق'},
                {'name': 'گلیسیرین', 'benefit': 'جذب رطوبت از هوا'},
                {'name': 'سرامیدها', 'benefit': 'ترمیم سد پوستی'}
            ],
            'oiliness': [
                {'name': 'نیاسینامید', 'benefit': 'کاهش چربی و بزرگ‌شدن منافذ'},
                {'name': 'روغن جوجوبا', 'benefit': 'تنظیم چربی پوست'},
                {'name': 'خاک رس', 'benefit': 'جذب چربی اضافی'}
            ]
        }
        
        result = []
        for issue in issues:
            if issue in ingredients:
                result.extend(ingredients[issue])
        
        # حذف تکراری‌ها
        unique_result = []
        seen = set()
        for item in result:
            if item['name'] not in seen:
                seen.add(item['name'])
                unique_result.append(item)
        
        return unique_result
    
    def _get_warnings(self, issues: List[str]) -> List[str]:
        """دریافت هشدارها"""
        warnings = []
        
        if 'acne' in issues:
            warnings.append('⚠️ در صورت جوش‌های التهابی به پزشک مراجعه کنید')
        
        if 'pigmentation' in issues:
            warnings.append('⚠️ از ضدآفتاب قوی استفاده کنید')
        
        if 'wrinkle' in issues:
            warnings.append('⚠️ مصرف رتینول را با دوز پایین شروع کنید')
        
        if 'sensitive' in issues:
            warnings.append('⚠️ از محصولات بدون عطر استفاده کنید')
        
        return warnings
    
    def get_consultation(self, 
                         user_age: Optional[int] = None,
                         user_gender: Optional[str] = None,
                         user_concern: Optional[str] = None) -> Dict:
        """
        دریافت مشاوره کامل
        
        Args:
            user_age: سن کاربر
            user_gender: جنسیت کاربر
            user_concern: نگرانی اصلی کاربر
        
        Returns:
            dict: مشاوره کامل
        """
        # تحلیل پوست
        analysis = self.analyze_skin(
            skin_type=None,
            issues=['acne', 'pigmentation'] if user_concern is None else [user_concern],
            age=user_age,
            gender=user_gender
        )
        
        # تولید پاسخ مشاوره
        return {
            'type': 'consultation',
            'title': '👨‍⚕️ مشاوره زیبایی شخصی‌سازی‌شده',
            'summary': f'بر اساس نوع پوست {analysis["skin_type"]["name"]} و مشکلات {", ".join([i["name"] for i in analysis["issues"]])}',
            'analysis': analysis,
            'next_steps': [
                '📋 روتین روزانه را شروع کنید',
                '🛒 محصولات پیشنهادی را تهیه کنید',
                '📅 بعد از ۴ هفته نتیجه را ارزیابی کنید',
                '👨‍⚕️ در صورت نیاز به پزشک مراجعه کنید'
            ]
        }


# ایجاد نمونه سراسری
consultation_service = ConsultationService()

# تست
if __name__ == "__main__":
    print("✅ ConsultationService ready")
    
    # تست مشاوره
    result = consultation_service.get_consultation(
        user_age=28,
        user_gender='female',
        user_concern='acne'
    )
    
    print(f"\n📊 Consultation Result:")
    print(f"   Title: {result['title']}")
    print(f"   Summary: {result['summary']}")
    print(f"   Skin Type: {result['analysis']['skin_type']['name']}")
    print(f"   Issues: {[i['name'] for i in result['analysis']['issues']]}")
    print(f"   Morning Routine: {result['analysis']['recommendations']['routine']['morning'][:3]}")