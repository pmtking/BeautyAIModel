from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class LanguageEntry:
    """ورودی چندزبانه برای هر عبارت"""
    
    persian: str = ''
    arabic: str = ''
    english: str = ''
    turkish: str = ''
    russian: str = ''
    kurdish: str = ''
    
    def get(self, lang: str = 'fa') -> str:
        """دریافت ترجمه به زبان مورد نظر"""
        mapping = {
            'fa': self.persian,
            'ar': self.arabic,
            'en': self.english,
            'tr': self.turkish,
            'ru': self.russian,
            'ku': self.kurdish,
        }
        return mapping.get(lang, self.english)
    
    def get_all(self) -> Dict[str, str]:
        """دریافت همه ترجمه‌ها"""
        return {
            'fa': self.persian,
            'ar': self.arabic,
            'en': self.english,
            'tr': self.turkish,
            'ru': self.russian,
            'ku': self.kurdish,
        }
    
    def has_translation(self, lang: str) -> bool:
        """بررسی وجود ترجمه برای یک زبان"""
        return bool(self.get(lang))


@dataclass
class BeautyTerm:
    """هر عبارت زیبایی با ترجمه‌های چندزبانه"""
    
    id: str
    translations: LanguageEntry
    category: str = ''
    description: Optional[str] = None


class MultilingualBeautySystem:
    """سیستم چندزبانه برای BeautyAI"""
    
    def __init__(self):
        self.terms: Dict[str, BeautyTerm] = {}
        self._init_terms()
    
    def _init_terms(self):
        """بارگذاری عبارات اولیه"""
        terms = [
            # اعمال
            BeautyTerm(
                id='smaller',
                translations=LanguageEntry(
                    persian='کوچک‌تر',
                    arabic='أصغر',
                    english='smaller',
                    turkish='daha küçük',
                    russian='меньше',
                    kurdish='piçûktir'
                ),
                category='action',
                description='کاهش اندازه و حجم'
            ),
            BeautyTerm(
                id='bigger',
                translations=LanguageEntry(
                    persian='بزرگ‌تر',
                    arabic='أكبر',
                    english='bigger',
                    turkish='daha büyük',
                    russian='больше',
                    kurdish='mezintir'
                ),
                category='action',
                description='افزایش اندازه و حجم'
            ),
            BeautyTerm(
                id='fuller',
                translations=LanguageEntry(
                    persian='پرتر',
                    arabic='أملىء',
                    english='fuller',
                    turkish='daha dolgun',
                    russian='полнее',
                    kurdish='tijtir'
                ),
                category='action',
                description='افزایش حجم و برجستگی'
            ),
            
            # نواحی
            BeautyTerm(
                id='nose',
                translations=LanguageEntry(
                    persian='بینی',
                    arabic='الأنف',
                    english='nose',
                    turkish='burun',
                    russian='нос',
                    kurdish='poz'
                ),
                category='area',
                description='ناحیه بینی'
            ),
            BeautyTerm(
                id='lip',
                translations=LanguageEntry(
                    persian='لب',
                    arabic='الشفاه',
                    english='lip',
                    turkish='dudak',
                    russian='губа',
                    kurdish='lêv'
                ),
                category='area',
                description='ناحیه لب'
            ),
            BeautyTerm(
                id='jaw',
                translations=LanguageEntry(
                    persian='فک',
                    arabic='الفك',
                    english='jaw',
                    turkish='çene',
                    russian='челюсть',
                    kurdish='çene'
                ),
                category='area',
                description='ناحیه فک'
            ),
            BeautyTerm(
                id='cheek',
                translations=LanguageEntry(
                    persian='گونه',
                    arabic='الخد',
                    english='cheek',
                    turkish='yanak',
                    russian='щека',
                    kurdish='rû'
                ),
                category='area',
                description='ناحیه گونه'
            ),
            BeautyTerm(
                id='eye',
                translations=LanguageEntry(
                    persian='چشم',
                    arabic='العین',
                    english='eye',
                    turkish='göz',
                    russian='глаз',
                    kurdish='çav'
                ),
                category='area',
                description='ناحیه چشم'
            ),
            
            # استایل‌ها
            BeautyTerm(
                id='heart_shape',
                translations=LanguageEntry(
                    persian='قلوه‌ای',
                    arabic='على شكل قلب',
                    english='heart shape',
                    turkish='kalp şekli',
                    russian='форма сердца',
                    kurdish='şiklê dil'
                ),
                category='style',
                description='فرم قلب با برجستگی مرکزی'
            ),
            BeautyTerm(
                id='slim_bridge',
                translations=LanguageEntry(
                    persian='قلمی',
                    arabic='رفیع',
                    english='slim bridge',
                    turkish='ince köprü',
                    russian='тонкий мост',
                    kurdish='pira zirav'
                ),
                category='style',
                description='پل بینی باریک و ظریف'
            ),
        ]
        
        for term in terms:
            self.terms[term.id] = term
    
    def get_term(self, term_id: str, lang: str = 'fa') -> Optional[str]:
        """دریافت یک عبارت به زبان مورد نظر"""
        term = self.terms.get(term_id)
        if term:
            return term.translations.get(lang)
        return None
    
    def get_by_category(self, category: str, lang: str = 'fa') -> Dict[str, str]:
        """دریافت همه عبارات یک دسته‌بندی"""
        result = {}
        for term_id, term in self.terms.items():
            if term.category == category:
                result[term_id] = term.translations.get(lang)
        return result
    
    def search(self, query: str, lang: str = 'fa') -> List[str]:
        """جستجوی عبارت در زبان مورد نظر"""
        results = []
        query_lower = query.lower()
        
        for term_id, term in self.terms.items():
            translation = term.translations.get(lang)
            if translation and query_lower in translation.lower():
                results.append(term_id)
        
        return results


# نمونه Singleton
multilingual = MultilingualBeautySystem()

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Testing Multilingual System")
    print("=" * 60)
    
    # تست با فارسی
    print("\n📝 Testing Persian (fa):")
    print(f"  smaller: {multilingual.get_term('smaller', 'fa')}")
    print(f"  nose: {multilingual.get_term('nose', 'fa')}")
    print(f"  heart_shape: {multilingual.get_term('heart_shape', 'fa')}")
    
    # تست با انگلیسی
    print("\n📝 Testing English (en):")
    print(f"  smaller: {multilingual.get_term('smaller', 'en')}")
    print(f"  nose: {multilingual.get_term('nose', 'en')}")
    
    # تست دسته‌بندی
    print("\n📝 Actions in Persian:")
    actions = multilingual.get_by_category('action', 'fa')
    for key, value in actions.items():
        print(f"  {key}: {value}")
    
    # تست جستجو
    print("\n📝 Search for 'بینی':")
    results = multilingual.search('بینی', 'fa')
    print(f"  Found: {results}")
    
    print("\n" + "=" * 60)
    print("✅ Test completed!")