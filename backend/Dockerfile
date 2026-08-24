# ============================================
# location : backend/Dockerfile
# ============================================

# استفاده از نسخه پایدار Python
FROM python:3.11-slim AS builder

WORKDIR /app

# ============================================
# ۱. نصب وابستگی‌های سیستمی
# ============================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================
# ۲. نصب وابستگی‌های Python
# ============================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ============================================
# ۳. مرحله نهایی (ساختار سبک‌تر)
# ============================================
FROM python:3.11-slim AS final

WORKDIR /app

# نصب وابستگی‌های اجرایی (بدون gcc/g++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# کپی وابستگی‌های نصب‌شده از مرحله builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# ============================================
# ۴. کپی کد پروژه
# ============================================
COPY . .

# ============================================
# ۵. ایجاد پوشه‌های مورد نیاز
# ============================================
RUN mkdir -p logs datasets uploads

# ============================================
# ۶. پورت و اجرا
# ============================================
EXPOSE 8000

CMD ["python", "-m", "app.main"]