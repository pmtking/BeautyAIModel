"""
🎯 FaceEdit API — ادیت حرفه‌ای با انتخاب کاربر
=================================================
POST /api/v1/face-edit/styles
  { image: base64, area: "nose"|"lip"|"jaw"|"cheek", style: "fantasy"|..., intensity: 0.7 }
  → { styles: [{ id, image, label, score }], area, style }

POST /api/v1/face-edit/apply
  { image: base64, style_id: 0, area, style }
  → { image: base64 }

GET  /api/v1/face-edit/areas
  → { areas: { nose: [...styles], lip: [...] } }
"""
import os, sys, base64, time, logging
from pathlib import Path

# engine path
ENGINE_DIR = str(Path(__file__).resolve().parent.parent.parent / 'engine')
sys.path.insert(0, ENGINE_DIR)

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import cv2
import numpy as np

router = APIRouter()
logger = logging.getLogger(__name__)

# موتور سراسری — lazy load
_engine = None
_variants_cache = {}  # session → variants


def get_engine():
    global _engine
    if _engine is None:
        try:
            from face_edit_engine import FaceEditEngine
            _engine = FaceEditEngine(device='cuda:0')
            _engine.load()
            logger.info('FaceEditEngine loaded')
        except Exception as e:
            logger.error(f'FaceEditEngine load failed: {e}')
            raise HTTPException(500, f'موتور بارگذاری نشد: {e}')
    return _engine


class StylesRequest(BaseModel):
    image: str                      # base64 JPEG
    area: str = 'nose'
    style: str = 'fantasy'
    intensity: float = 0.7
    n_variants: int = 3


class ApplyRequest(BaseModel):
    image: str                      # base64 JPEG (اصلی)
    style_id: int = 0
    area: str = 'nose'
    style: str = 'fantasy'
    intensity: float = 0.7


def b64_to_img(s: str) -> np.ndarray:
    if s.startswith('data:'):
        s = s.split(',', 1)[1]
    data = base64.b64decode(s)
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, 'تصویر نامعتبر')
    return img


def img_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()


@router.get('/face-edit/areas')
async def get_areas():
    from face_edit_engine import FaceEditEngine
    areas = {}
    for area, styles in FaceEditEngine.AREA_STYLES.items():
        areas[area] = [{'id': k, 'label': v[0]} for k, v in styles.items()]
    return {'status': 'success', 'areas': areas}


@router.post('/face-edit/styles')
async def generate_styles(req: StylesRequest):
    """چند استایل مختلف بساز تا کاربر انتخاب کنه"""
    t0 = time.time()
    try:
        engine = get_engine()
        image = b64_to_img(req.image)
        variants = engine.generate_styles(
            image, req.area, req.style,
            intensity=req.intensity,
            n_variants=req.n_variants,
        )
        if not variants:
            return JSONResponse({'status': 'error', 'message': 'تولید ممکن نبود'}, 400)
        # حافظه: variants + عکس اصلی برای apply
        cache_key = f'{id(req.image)}'
        _variants_cache[cache_key] = (image, variants)
        # محدود کردن حافظه
        while len(_variants_cache) > 20:
            _variants_cache.pop(next(iter(_variants_cache)))
        
        return {
            'status': 'success',
            'styles': [
                {
                    'id': v['id'],
                    'image': img_to_b64(v['image']),
                    'label': v['label'],
                    'score': v['score'],
                }
                for v in variants
            ],
            'area': req.area,
            'style': req.style,
            'elapsed': round(time.time() - t0, 1),
        }
    except Exception as e:
        logger.error(f'styles error: {e}')
        return JSONResponse({'status': 'error', 'message': str(e)}, 500)


@router.post('/face-edit/apply')
async def apply_choice(req: ApplyRequest):
    """استایل انتخابشده کاربر رو اعمال کن"""
    try:
        engine = get_engine()
        image = b64_to_img(req.image)
        variants = engine.generate_styles(
            image, req.area, req.style, intensity=req.intensity,
        )
        if not variants:
            return JSONResponse({'status': 'error', 'message': 'تولید ممکن نبود'}, 400)
        result = engine.apply_choice(image, req.style_id, variants)
        if result is None:
            return JSONResponse({'status': 'error', 'message': 'استایل نامعتبر'}, 400)
        
        return {
            'status': 'success',
            'image': img_to_b64(result),
            'area': req.area,
            'style': req.style,
            'variant_id': req.style_id,
        }
    except Exception as e:
        logger.error(f'apply error: {e}')
        return JSONResponse({'status': 'error', 'message': str(e)}, 500)
