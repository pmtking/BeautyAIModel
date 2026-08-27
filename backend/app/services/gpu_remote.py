"""
📡 GPU Remote Client — پل به کیس ویندوزی خانه
==============================================
اگر GEN_API_URL تنظیم باشد، ادیت‌های سنگین (نیم‌رخ/شدت بالا) به GPU Worker
خانه فرستاده میشود؛ در دسترس نبودنِ آن → خودکار برگشت به موتور آناتومیک.

تنظیمات (متغیر محیطی):
    GEN_API_URL=http://192.168.1.38:8001     # فعال
    GEN_API_URL=                              # غیرفعال (پیشفرض)
    GEN_TIMEOUT_S=120                         # اختیاری

تست سریع از ترمینال:
    python -m backend.app.services.gpu_remote
"""
import os
import time
import base64
import logging

import numpy as np

logger = logging.getLogger(__name__)

GEN_API_URL = os.environ.get('GEN_API_URL', '').rstrip('/')
GEN_TIMEOUT = float(os.environ.get('GEN_TIMEOUT_S', '120'))
# 🔐 اگر ورکر با GPU_WORKER_TOKEN اجرا شده، همین مقدار را اینجا هم بگذار
GPU_TOKEN = os.environ.get('GPU_WORKER_TOKEN', '')
_AUTH_HEADERS = {'X-Token': GPU_TOKEN} if GPU_TOKEN else {}


def is_configured() -> bool:
    return bool(GEN_API_URL)


def health(timeout: float = 4.0):
    """سلامت ورکر — None یعنی در دسترس نیست."""
    if not is_configured():
        return None
    import urllib.request, json
    try:
        req = urllib.request.Request(f'{GEN_API_URL}/health',
                                     headers=_AUTH_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        logger.info(f'gpu worker unreachable: {e}')
        return None


def generate_edit(image_bgr, action: str, intensity: float):
    """ادیت مولد روی GPU خانه. خروجی BGR یا raise RuntimeError."""
    if not is_configured():
        raise RuntimeError('GEN_API_URL تنظیم نشده')
    import cv2
    import urllib.request, json as _json

    ok, enc = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError('encode failed')

    payload = _json.dumps({
        'image': base64.b64encode(enc.tobytes()).decode(),
        'action': action,
        'intensity': float(intensity),
    }).encode()

    req = urllib.request.Request(
        f'{GEN_API_URL}/generate', data=payload,
        headers={'Content-Type': 'application/json', **_AUTH_HEADERS},
        method='POST')
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=GEN_TIMEOUT) as r:
            resp = _json.loads(r.read().decode())
    except Exception as e:
        raise RuntimeError(f'GPU worker error: {e}')

    if resp.get('status') != 'success':
        raise RuntimeError(resp.get('message', 'worker failed'))

    buf = np.frombuffer(base64.b64decode(resp['image']), np.uint8)
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError('worker returned invalid image')
    logger.info(f'⚡ remote gen ok in {time.time()-t0:.1f}s ({resp.get("engine")})')
    return out


# ─────────────────────────────────────────────
if __name__ == '__main__':
    print(f'GEN_API_URL = {GEN_API_URL or "(خالی — غیرفعال)"}')
    h = health()
    if h is None:
        print('❌ worker در دسترس نیست')
    else:
        print(f'✅ worker: {h}')
