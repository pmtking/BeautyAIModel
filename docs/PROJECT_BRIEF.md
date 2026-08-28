# PROJECT BRIEF & ROADMAP — BeautyAIModel (BUTI)
# نگهداری‌شده در کنار پروژه — این فایل «حافظه» ماست.
# تاریخ آخرین به‌روزرسانی: 2026-08-28

## وضعیت کلی
پروژه: پلتفرم هوش مصنوعی زیبایی BUTI
- موتور شبیه‌سازی/تغییر چهره (بینی/لب/گونه/فک/...)
- چت‌بات فارسی «بوتی» (مشاور زیبایی)
- آپلود دیتاست برای بهبود مولد تصویر (دادن به مطب‌های زیبایی)

## چه چیزی آماده است (بک‌اند)
✅ FastAPI با روت‌ها: edit، chat، dataset، face_edit، three_d، retouch، manual_edit، avatar_3d، analyze
✅ /api/v1/dataset — ثبت‌نام، لاگین، ساخت کاربر ادمین، آپلود دیتاست (۶ نما قبل/بعد)، لیست، ویرایش، حذف
✅ دیتای JSON در backend/app/data/ (users.json + datasets.json + datasets/)
✅ چت‌بات v3 — بازنویسی شد (تشخیص نیت وزن‌دار + حافظه + دانش قیمت/ناحیه + LLM اختیاری)
✅ تست چت: ۱۲ تا ۱۴ حالت نیت درست کار می‌کند
✅ /mirror — دموی آینه هوشمند (موبایل)
مصرف API: /api/v1/edit (فایل + text + intensity) ، /api/v1/chat (user_id + text + has_photo)

## چه چیزی باید ساخته شود (کار فعلی/بعدی)
🔲 [REK] داشبورد/سایت حرفه‌ای و شکیل — صفحه اصلی + Login + آپلود دیتاست + مدیریت کاربر + گالری دیتاست
🔲 [REK] بخش چت با آپلود عکس برای تست
🔲 [REK] صفحه‌ی Landing / معرفی محصول (برای تحویل به کلینیک‌ها)
🔲 پنل ادمین: آمار، نمودار، مدیریت کاربر، مدیریت دیتاست
🔲 اتصال موتور قوی‌تر (سیستم خانه) به‌عنوان GPU/LLM ریموت
🔲 بهبود بیشتر مولد تصویر (کیفیت/دقت)

## نحوه اجرا
cd backend && source ../venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
(یا python app/main.py — reload=True)

## اعتبار ادمین پیش‌فرض (dataset_api)
admin / butiadmin2026  (از env قابل تغییر)

## نکته مهم
- Don't lose: backend/app/api/v1/dataset_api.py (core admin API)
- دیتا: backend/app/data/
- استاتیک/صفحات: backend/app/static/
