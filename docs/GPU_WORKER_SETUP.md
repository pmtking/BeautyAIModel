# 🖥️ راهنمای راه‌اندازی GPU Worker روی کیس ویندوزی خانه
**پروژه:** BeautyAI / BUTI — موتور مولد SDXL + ControlNet
**سخت‌افزار هدف:** i3 نسل ۱۳ + RAM 16GB + RTX 12GB (ویندوز)
**آخرین به‌روزرسانی:** ۲۵ اوت ۲۰۲۶

---

## 📋 چی ساخته شد (روی لپتاپ، آماده است)

| فایل | نقش |
|------|-----|
| `backend/gpu_worker.py` | سرور مستقل که روی کیس ویندوز اجرا میشود — لود SDXL و تولید |
| `backend/app/services/gpu_remote.py` | کلاینت داخل بک‌اند لپتاپ — درخواست را برای خانه میفرستد |
| `backend/app/api/v1/edit.py` | ⚡ جدید: اگر درخواست «سنگین» بود و ورکر زنده → خودکار ریدایرکت |
| `backend/app/services/generative.py` | فیکس شد: سقف VRAM از ‎۱۴→۱۰GB‏ (کارت ۱۲گیگ قبول شد) |

منطق مسیریابی: **ادیت ملایم → آناتومیک لوکال (~۳ ثانیه)**
**نیم‌رخ / شدت ≥۷۵٪ / اکشن سنگین → GPU خانه (~۲۰-۴۰ ثانیه)**
**ورکر خاموش → خودکار برگشت به آناتومیک (اپ هرگز نمی‌شکند)**

---

## 🚀 مراحل روی کیس ویندوز (شب، پشت سر هم)

### ۱) نصب پایه (فقط بار اول) — PowerShell با دسترسی Admin
```powershell
# پروژه را از طریق گیت/فلش/لن به D:\BeautyAIModel برسان
cd D:\BeautyAIModel
python -m venv venv-gpu
.\venv-gpu\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install fastapi uvicorn pydantic numpy opencv-python pillow diffusers transformers accelerate safetensors
```
> ⚠️ دانلود اولیه مدل‌ها (~۱۲GB) بار اولِ اولین تولید انجام میشود. اگر HF کند بود:
> `set HF_ENDPOINT=https://hf-mirror.com`

### ۲) تست بدون مدل (زنجیره ارتباط)
```powershell
cd D:\BeautyAIModel\backend
set WORKER_MOCK=1
python gpu_worker.py
```
از مرورگر کیس: `http://localhost:8001/health` → باید `"mock": true, "model_loaded": true` ببینی.

### ۳) اجرای واقعی GPU
```powershell
cd D:\BeautyAIModel\backend
python gpu_worker.py        # بار اول ~۶۰ثانیه لود؛ بگذار بالا بیاید
```
چک: `http://localhost:8001/health` → باید اسم کارت (`NVIDIA ...`) را نشان دهد.

### ۴) باز کردن پورت برای شبکه (Admin)
```powershell
netsh advfirewall firewall add rule name="BeautyAI GPU Worker" dir=in action=allow protocol=TCP localport=8001
```

### ۵) پیدا کردن IP کیس
```powershell
ipconfig    # IPv4 مثل 192.168.1.xx یادداشت کن
```

---

## 🔌 سمت لپتاپ (توسعه)

### همان WiFi خونه:
```bash
cd ~/Desktop/project/BeautyAIModel/backend
GEN_API_URL=http://192.168.1.XX:8001 ../venv/bin/python -m backend.app.services.gpu_remote
# باید ✅ worker: {...} ببینی — بعد سرور اصلی را همینطور بالا بیاور:
GEN_API_URL=http://192.168.1.XX:8001 ../venv/bin/python run.py
```

### از بیرون خانه (دموی سرمایه‌گذار) — Tailscale:
```bash
# روی هر دو دستگاه نصب و لاگین با یک اکانت:
sudo apt install tailscale        # لپتاپ (ویندوز: از tailscale.com)
tailscale up                      # IP مجازی 100.x.y.z میدهد
# سپس:
GEN_API_URL=http://100.x.y.z:8001 ...
```
مزیت: بدون port forwarding، رمزنگاری WireGuard، IP ثابت — حتی با اینترنت ایران جواب داده.

---

## 🧪 چکلیست صحت (به ترتیب)

```bash
# ۱) ورکر زنده؟
curl http://192.168.1.XX:8001/health

# ۲) تولید واقعی روی ورکر؟ (عکس را base64 کن یا از اسکریپت زیر)
python -m backend.app.services.gpu_remote

# ۳) مسیر کامل از API اصلی؟
curl -s -X POST http://localhost:8000/api/v1/edit \
  -F "file=@test_images/test.jpg" \
  -F "text=نوک بینی رو خیلا بالا ببر" \
  | jq '.data.engine'     # باید "remote-gpu" شود
```
✅ نتیجه موفق = `engine: remote-gpu` + `jobs_done` در health زیاد شود.
⚠️ اگر `anatomic-fallback` دیدی یعنی ورکر وسط کار خطا داد → لاگ ورکر را بخوان.

---

## 🐛 مشکلات رایج

| علامت | علت | حل |
|-------|-----|-----|
| `torch.cuda.is_available() = False` | PyTorch نسخه CPU است | دوباره با `--index-url .../whl/cu121` نصب کن |
| health میرسد ولی generate تایم‌اوت | اولین لود مدل هنوز تمام نشده | `/warmup` صدا بزن و صبر کن |
| از لپتاپ unreachable، از خود کیس OK | فایروال ویندوز | مرحله ۴ بالای صفحه |
| OOM حین تولید | رمز کارت پر | `set GEN_CANDIDATES=1` و مطمئن شو چیز دیگری VRAM نگرفته |

---

## 💰 بعداً (اختیاری): همیشه-روشن بودن
وقتی دمو تمام شد، اگر میخواهی ورکر ۲۴/۷ باشد:
- Task Scheduler ویندوز → اجرای `gpu_worker.py` هنگام Startup
- یا تبدیل به سرویس با [NSSM](https://nssm.cc)

---
*ساختهشده توسط ox-alpha — جلسه کاری ۲۵ اوت ۲۰۲۶*
