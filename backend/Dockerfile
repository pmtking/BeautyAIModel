# syntax=docker/dockerfile:1
# ============================================================
#  BeautyAI Backend — Fast, CPU-only, production-ready
#  ⚠️ build context = ریشه پروژه (نه پوشه backend)
#
#  Build:  docker build -f backend/Dockerfile -t beautyai-backend .
#  Run:    docker run -p 8000:8000 beautyai-backend
# ============================================================
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# --- فقط کتابخانه‌های سیستمی لازم برای OpenCV/MediaPipe ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# --- لایه پایتون: اول requirements تا کش بشه ---
COPY backend/docker-requirements.txt .
RUN pip install --timeout 120 --retries 5 -r docker-requirements.txt

# --- کد برنامه ---
COPY backend/run.py backend/run_docker.py ./
COPY backend/app/ ./app/
COPY backend/engine/ ./engine/
COPY backend/websocket/ ./websocket/

# --- موتور AI (پوشه ai_training در ریشه پروژه) ---
COPY ai_training/ ./ai_training/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# بدون reload برای بالا آمدن سریع؛ workers=1 چون موتور AI singleton است
CMD ["python", "run_docker.py"]
