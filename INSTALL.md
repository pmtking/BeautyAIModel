# 🚀 راهنمای نصب و اجرای BUTI Backend

## ۱. نصب (یک‌بار)

```bash
# ساخت محیط مجازی
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## ۲. اجرا (روی سرور)

```bash
cd backend
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

سپس در مرورگر باز کنید:
- سایت اصلی: `http://localhost:8000/`
- مستندات API: `http://localhost:8000/docs`
- صفحه آینه (موبایل): `http://localhost:8000/mirror`

## ۳. لاگین ادمین پیش‌فرض
- نام کاربری: `admin`
- رمز عبور: `butiadmin2026`

## ۴. اتصال LLM ریموت (اختیاری، برای چت هوشمندتر)
```bash
export BEAUTY_LLM_URL="http://192.168.1.x:11434/v1"   # آدرس vLLM/Ollama
export BEAUTY_LLM_MODEL="qwen2.5"  # یا هر مدل دلخواه
export BEAUTY_LLM_KEY=""   # اگر نیاز به کلید است
```
سپس سرور را ری‌استارت کنید. در غیر این صورت، موتور محلی (کاملاً هوشمند فارسی) کار می‌کند.

## ۵. اتصال GPU ریموت (اختیاری، برای مولد تصویر قوی‌تر)
در `backend/app/services/gpu_remote.py` آدرس سرور GPU خانه را تنظیم کنید. در حال حاضر موتور پیش‌فرض BeautyEngine (CPU) کار می‌کند.

## ۶. دایرکتوری‌های مهم
- `backend/app/static/index.html` — سایت اصلی (لندینگ + داشبورد + چت + آپلود + ادمین)
- `backend/app/static/live_mirror.html` — دموی آینه (موبایل)
- `backend/app/data/users.json` — کاربران
- `backend/app/data/datasets.json` — متادیتای دیتاست‌ها
- `backend/app/data/datasets/` — فایل‌های عکس
- `backend/app/services/chat_bot.py` — موتور چت فارسی (نسخه ۳)
- `backend/app/api/v1/dataset_api.py` — API دیتاست + احراز هویت
- `backend/app/api/v1/edit.py` — API مولد تصویر
- `docs/PROJECT_BRIEF.md` — نقشه راه پروژه
