# 🧴 سیستم مشاوره پوست BeautyAI — معماری علمی
## بر اساس: Baumann Skin Typing + Haut.AI + NAOS Skin Observer

---

## ۱. تایپ‌سازی پوست — سیستم بامان (۱۶ نوع)

۴ پارامتر دوتایی → ۱۶ ترکیب:

| محور | قطب‌ها |
|------|--------|
| **O/D** آب‌رسانی | Oily (چرب) / Dry (خشک) |
| **S/R** حساسیت | Sensitive / Resistant |
| **P/N** رنگدانه | Pigmented (لک‌دار) / Non-pigmented |
| **W/T** چروک | Wrinkled / Tight |

مثلاً `OSPW` = چرب، حساس، لک‌دار، چروک

### پرسشنامه هوشمند (۱۲ سوال کلیدی):
1. بعد از شستشو پوستم کشیده میشه؟ (O/D)
2. آکنه/جوش مکرر دارم؟ (O)
3. اگزما/سوزش از محصولات؟ (S/R)
4. لک حاملگی/آفتاب؟ (P)
5. خط خنده/اخم دائم؟ (W)
6. سن (تقریبی) — برای اولویت‌بندی W
...

### تشخیص تصویری (CNN):
- **آکنه:** YOLOv8-face-acne یا مدل fine-tune شده
- **چروک:** edge-density در نواحی پیشانی/چشم
- **منافذ:** local-variance روی گونه‌ها
- **لک:** color-segmentation در فضای LAB
- **قرمزی:** Erythema index = a* channel

---

## ۲. روتین شخصی‌سازی‌شده (۴۰,۰۰۰ ترکیب)

### ستون‌های روتین (صبح/شب):

| مرحله | صبح | شب |
|-------|-----|-----|
| ۱. پاک‌کننده | ملایم بر اساس O/D | دوبل (روغن+فوم اگر آرایش) |
| ۲. تونر | ترمیم pH | AHA/BHA اگر منافذ باز |
| ۳. سرم | ویتامین C (اگر P) | هیالورونیک (D) یا نیاسینامید (O) |
| ۴. مرطوب‌کننده | سبک ژلی (O) / غنی (D) | ترمیمی با سرامید |
| ۵. ضدآفتاب | SPF50+ PA++++ الزامی | — |
| ۶. درمان | — | رتینول (W، تدریجی) |

### قواعد تداخل:
- رتینول × ویتامین C → شب/روز جدا
- AHA/BHA × رتینول → یک شب در میان
- نیاسینامید با همه سازگار ✅

---

## ۳. پیاده‌سازی فنی در BeautyAI

```
backend/app/services/skin_advisor.py
├── class BaumannQuiz        # ۱۲ سوال → ۴ حرف
├── class SkinImageAnalyzer  # CNN تحلیل عکس
│   ├── detect_acne()
│   ├── estimate_wrinkles()
│   ├── measure_pores()
│   └── spot_pigmentation()
├── class RoutineBuilder
│   └── build(baumann_type, issues[], age, budget)
└── class ProductRecommender  # پایگاه محصول ایرانی/بین‌المللی
```

### API endpoint جدید:
```
POST /api/v1/skin/advisor
{
  "image": base64,
  "quiz_answers": {...},
  "age": 28,
  "budget": "mid"
}
→ {
  "skin_type": "OSPW",
  "issues": [
    {"type":"acne","confidence":0.82,"region":"forehead"},
    {"type":"dark_spots","confidence":0.71,"region":"cheeks"}
  ],
  "routine": {
    "morning": [...],
    "evening": [...]
  },
  "products": [...],
  "warnings": ["رتینول را هفته‌ای ۲ بار شروع کن"]
}
```

## ۴. ادغام با چت‌بات
کاربر: «پوستم جوش داره چه کنم؟»
چت‌بات: تشخیص intent=skin → درخواست سلفی + پرسشنامه کوتاه
       → تحلیل → روتین + توضیح علمی هر قدم

## ۵. مرجع مقالات ذخیره‌شده
- `/tmp/nose_article.txt` — انواع عمل بینی (ritapezeshkan)
- `/tmp/inject_article.txt` — انواع تزریق زیبایی (bartarinha)
- `/tmp/hifu_article.txt` — هایفوتراپی (doctoreto)
- `/tmp/ghabghab_article.txt` — غبغب (pezeshk24)
- Haut.AI — ۱۵۰+ بیومارکر، آموزش ۳M تصویر (الگوی ما)
