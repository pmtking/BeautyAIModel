#!/usr/bin/env bash
# ============================================================
#  BeautyAI — اجرای جلسه (یک دستور، همه چیز)
#  HTTPS برای دسترسی دوربین از موبایل
#
#  استفاده:  ./start_demo.sh
# ============================================================
set -e
cd "$(dirname "$0")"

echo "🚀 BeautyAI — راه‌اندازی دموی زنده..."

# ---------- گواهی خودامضا (برای دوربین موبایل لازم است) ----------
mkdir -p .certs
if [ ! -f .certs/cert.pem ]; then
  echo "🔐 ساخت گواهی SSL..."
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout .certs/key.pem -out .certs/cert.pem \
    -days 365 -subj "/CN=beautyai-local" >/dev/null 2>&1
fi

# ---------- پورت قبلی را آزاد کن ----------
fuser -k 8000/tcp >/dev/null 2>&1 || true
sleep 1

source venv/bin/activate

IP=$(hostname -I | awk '{print $1}')

echo ""
echo "=============================================="
echo " ✅ آماده‌ی جلسه!"
echo ""
echo " 📱 روی گوشی باز کن:"
echo "    https://$IP:8000/mirror"
echo ""
echo " ⚠️  اولین بار مرورگر هشدار می‌دهد:"
echo "    «Advanced» ← «Proceed to ... (unsafe)» را بزن"
echo "    (این پیام طبیعی است — گواهی محلی است)"
echo ""
echo " 💻 Swagger لپ‌تاپ:  https://localhost:8000/docs"
echo " ⛔ خاموش کردن:  Ctrl+C"
echo "=============================================="
echo ""

cd backend
python - << EOF
import uvicorn
uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    ssl_certfile="../.certs/cert.pem",
    ssl_keyfile="../.certs/key.pem",
)
EOF
