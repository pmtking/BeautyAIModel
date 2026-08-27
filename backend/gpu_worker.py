"""
🧊 GPU Worker — اجرا روی کیس ویندوزی خانه (RTX 12GB)
====================================================
سرور مستقل FastAPI که فقط کار «تولید مولد» را انجام میدهد.
لپتاپ توسعه از طریق GEN_API_URL به این سرور وصل میشود.

اجرا روی ویندوز (PowerShell در پوشه backend):
    .\\venv-gpu\\Scripts\\activate
    set GEN_CANDIDATES=1
    python gpu_worker.py            → روی پورت 8001

حالت تست بدون GPU (روی هر ماشینی):
    set WORKER_MOCK=1
    python gpu_worker.py            → خروجی ساختگی برای تست زنجیره
"""
import os
import io
import sys
import time
import base64
import logging
import threading

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
log = logging.getLogger('gpu-worker')

MOCK = os.environ.get('WORKER_MOCK', '') == '1'
N_CANDIDATES = int(os.environ.get('GEN_CANDIDATES', '1'))
PORT = int(os.environ.get('GPU_WORKER_PORT', '8001'))

# ─────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title='BeautyAI GPU Worker', version='1.0')
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_methods=['*'], allow_headers=['*'],
)


class GenReq(BaseModel):
    image: str                 # base64 JPEG (بدون پیشوند data:)
    action: str = 'ideal_realistic'
    intensity: float = 0.7


_state = {
    'editor': None,          # ProNoseEditor
    'ns': None,              # namespace ماژول beautygen
    'loading': False,
    'load_error': None,
    'busy': False,
    'started': time.time(),
    'done': 0,
}

# 🔐 توکن اختیاری ولی شدیداً توصیه‌شده وقتی ورکر از بیرون خانه در دسترس است
#    ویندوز:  set GPU_WORKER_TOKEN=yek-ramz-boland
WORKER_TOKEN = os.environ.get('GPU_WORKER_TOKEN', '')


def _auth_ok(request):
    """اگر توکن تنظیم شده باشد، هدر X-Token باید مطابقت داشته باشد."""
    if not WORKER_TOKEN:
        return True
    return request.headers.get('x-token') == WORKER_TOKEN


def gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {'gpu': None, 'vram_gb': 0.0}
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory / 10**9
        return {'gpu': name, 'vram_gb': round(total, 1)}
    except Exception:
        return {'gpu': None, 'vram_gb': 0.0}


def load_model():
    """لود یکباره NoseGenEngine (SDXL+ControlNet+کامپوزیت قفل‌هویت)."""
    if _state['editor'] is not None or _state['loading']:
        return
    _state['loading'] = True
    try:
        info = gpu_info()
        if not info['gpu']:
            raise RuntimeError('CUDA در دسترس نیست — درایور/PyTorch-CUDA را چک کن')
        log.info(f'🔥 loading SDXL on {info["gpu"]} ({info["vram_gb"]}GB) …')

        # 🎯 موتور حرفه‌ای: crop-to-face + کامپوزیت تضمینی فقط-بینی
        from nose_gen_engine import NoseGenEngine
        eng = NoseGenEngine()
        eng.load()
        _state['editor'] = eng
        log.info('✅ model ready (NoseGenEngine v3)')
    except Exception as e:
        _state['load_error'] = str(e)
        log.error(f'model load FAILED: {e}')
    finally:
        _state['loading'] = False


@app.get('/health')
def health(request: Request):
    if not _auth_ok(request):
        raise HTTPException(401, 'token namotabar')
    info = gpu_info()
    return {
        'status': 'ok' if (MOCK or _state['editor']) else ('loading' if _state['loading'] else 'idle'),
        'mock': MOCK,
        'model_loaded': MOCK or _state['editor'] is not None,
        'load_error': _state['load_error'],
        'busy': _state['busy'],
        'gpu': info['gpu'] or ('MOCK' if MOCK else None),
        'vram_gb': info['vram_gb'],
        'candidates_per_job': N_CANDIDATES,
        'jobs_done': _state['done'],
        'uptime_s': round(time.time() - _state['started']),
    }


@app.post('/warmup')
def warmup(request: Request):
    """لود مدل — اولین بار ~۶۰ثانیه + دانلود ~۱۲GB"""
    if not _auth_ok(request):
        raise HTTPException(401, 'token namotabar')
    if MOCK:
        return {'status': 'ok', 'message': 'mock mode'}
    t0 = time.time()
    load_model()
    if _state['editor'] is None:
        raise HTTPException(500, _state['load_error'] or 'load failed')
    return {'status': 'ok', 'elapsed_s': round(time.time() - t0, 1)}


@app.post('/generate')
def generate(req: GenReq, request: Request):
    if not _auth_ok(request):
        raise HTTPException(401, 'token namotabar')
    if _state['busy']:
        raise HTTPException(503, 'worker busy — یک لحظه بعد تلاش کن')
    buf = np.frombuffer(base64.b64decode(req.image), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, 'تصویر نامعتبر')

    _state['busy'] = True
    t0 = time.time()
    try:
        if MOCK:
            out = _mock_edit(img, req.action)
        else:
            load_model()
            ed = _state['editor']
            if ed is None:
                raise HTTPException(500, _state['load_error'] or 'model not loaded')
            if N_CANDIDATES <= 1:
                out, meta = ed.edit(img, req.action, req.intensity)
                log.info(f'gen meta: {meta}')
            else:
                best, best_meta = None, {}
                for _ in range(N_CANDIDATES):
                    cand, m = ed.edit(img, req.action, req.intensity)
                    sim = m.get('identity_sim')
                    if best is None or (sim or 0) > (best_meta.get('identity_sim') or 0):
                        best, best_meta = cand, m
                out = best
        _state['done'] += 1
        _, jpg = cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 93])
        return {
            'status': 'success',
            'image': base64.b64encode(jpg.tobytes()).decode(),
            'engine': 'mock' if MOCK else 'sdxl-controlnet',
            'elapsed_s': round(time.time() - t0, 1),
        }
    finally:
        _state['busy'] = False


def _to_bgr(pil_img, shape):
    arr = np.array(pil_img)[:, :, ::-1]          # RGB→BGR
    if arr.shape[:2] != shape[:2]:
        arr = cv2.resize(arr, (shape[1], shape[0]),
                         interpolation=cv2.INTER_LANCZOS4)
    return np.ascontiguousarray(arr)


def _mock_edit(img, action):
    """خروجی ساختگی قابل تشخیص — برای تست زنجیره بدون GPU"""
    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.circle(overlay, (w // 2, int(h * 0.45)), int(min(h, w) * 0.10),
               (0, 140, 255), -1)
    out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    cv2.putText(out, f'GPU-WORKER-MOCK:{action}', (20, h - 24),
                cv2.FONT_HERSHEY_SIMPLEX, min(w / 900, 1.4), (0, 255, 255), 2)
    return out


if __name__ == '__main__':
    import uvicorn
    log.info(f'GPU Worker starting on :{PORT}  (mock={MOCK})')
    uvicorn.run(app, host='0.0.0.0', port=PORT, log_level='info')
