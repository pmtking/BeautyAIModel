# 🧠 BeautyGen — آموزش مدل مولد خودت (گام‌به‌گام)

**دیتاست آماده:** ۳,۵۹۲ جفت قبل/بعد لیبل‌دار (۳۳۷۷ train)
**فایل آماده آپلود:** `/tmp/beautygen_data.tar.gz` (115MB)

## ⏱ زمان لازم: ~۱.۵ ساعت تا اولین مدل تست

---

## گام ۱ — دیتاست را بفرست GPU (۵ دقیقه)

```bash
# فایل آماده است؛ فقط آپلود کن
# روش A: Kaggle (رایگان، 30h GPU در هفته) ← پیشنهاد من
#   kaggle.com → Datasets → New → آپلود beautygen_data.tar.gz

# روش B: Google Drive
#   فایل را در Drive بگذار → در Colab mount کن
```

## گام ۲ — Colab notebook (۴۵-۶۰ دقیقه آموزش)

1. برو به `colab.research.google.com`
2. Runtime → Change runtime → **T4 GPU**
3. سلول‌های `notebooks/train_beautygen.py` را به ترتیب اجرا کن:

```
CELL 1: نصب کتابخانه‌ها (~2 دقیقه)
CELL 2: آماده‌سازی دیتاست (~3 دقیقه)
CELL 3: آموزش LoRA (~45 دقیقه روی T4)
```

4. خروجی: `beuti-nose-lora/` (~50MB) — دانلود کن

## گام ۳ — وصل کردن به سرور (۱۰ دقیقه)

فایل LoRA را اینجا کپی کن:
```bash
beuti-nose-lora/
├── adapter_config.json
└── adapter_model.safetensors
```

سپس در سرور با GPU:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install diffusers peft accelerate transformers
systemctl restart beautyai  # موتور هیبرید خودکار فعال میشود!
```

## 🎛 چطور کار میکند (بعد از فعال‌سازی)

| درخواست | مسیر | زمان | کیفیت |
|---------|------|------|-------|
| «بینی کمی باریک» فرانت | آناتومیک فعلی | ۶s | خوب |
| «نوک سر بالا» شدت زیاد | **مدل مولد** | ۱۵s | فوق‌واقعی ⭐ |
| هر تغییری در نیم‌رخ | **مدل مولد** | ۱۵s | فوق‌واقعی ⭐ |

روتر هیبرید (`hybrid_router.py`) خودش تصمیم میگیرد.

## 💡 نکته مهم برای دمو سرمایه‌گذار

حتی قبل از آموزش کامل، میتوانی از **SD inpainting عمومی** استفاده کنی:
- بدون LoRA هم کیفیت بهتر از warp دستی است
- بعد که LoRA خودت آمد، swap میکنی و همه‌چیز بهتر میشود

## ❓ اگر GPU نداری اصلاً

RunPod: `rtx-4090` ساعتی ~0.35$ → آموزش کامل < 2$
یا از API آماده: Replicate / fal.ai (~0.02$ به ازای هر عکس)
