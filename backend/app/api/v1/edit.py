# app/api/v1/edit.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import sys
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import time
import logging

logger = logging.getLogger(__name__)

# ============================================
# ✅ ایجاد router
# ============================================
router = APIRouter()

# ============================================
# ✅ تنظیم مسیر به مدل‌های ai_training
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
AI_TRAINING_DIR = BASE_DIR / "ai_training"

if not AI_TRAINING_DIR.exists():
    import os
    cwd = Path(os.getcwd())
    AI_TRAINING_DIR = cwd / "ai_training"
    if not AI_TRAINING_DIR.exists():
        raise RuntimeError(f"پوشه ai_training در هیچ‌کدام از مسیرهای {BASE_DIR} و {cwd} پیدا نشد!")

sys.path.insert(0, str(AI_TRAINING_DIR))
sys.path.insert(0, str(AI_TRAINING_DIR / "src" / "models"))

# ============================================
# ✅ import مدل
# ============================================
try:
    from beauty_engine.model import beauty_engine
    logger.info("✅ beauty_engine loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to load beauty_engine: {e}")
    beauty_engine = None


@router.post("/edit")
async def edit_face(
    file: UploadFile = File(..., description="تصویر صورت"),
    text: str = Form(..., description="درخواست به زبان فارسی"),
    intensity: float = Form(None, description="شدت تغییر (اختیاری، ۰.۱ تا ۱.۰)")
):
    start_time = time.time()

    try:
        if beauty_engine is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "message": "مدل در دسترس نیست",
                    "processing_time": round(time.time() - start_time, 3)
                }
            )

        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="فایل باید تصویر باشد")

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="تصویر معتبر نیست")

        result = beauty_engine.process(image, text, intensity)

        if result.get('status') == 'error':
            return JSONResponse(
                status_code=400,
                content={
                    "status": "error",
                    "message": result.get('message', 'خطا در پردازش'),
                    "processing_time": round(time.time() - start_time, 3)
                }
            )

        _, buffer = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "data": {
                "image": img_base64,
                "description": result.get('description', ''),
                "changes": result.get('changes', {}),
                "intensity": result.get('intensity', 0.5),
                "message": result.get('message', '✅ تغییرات با موفقیت اعمال شد')
            },
            "processing_time": round(time.time() - start_time, 3)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Edit error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "processing_time": round(time.time() - start_time, 3)
            }
        )


@router.get("/edit/styles")
async def get_styles(area: str = "lip"):
    styles = {
        'lip': [
            {'id': 'heart_shape', 'name': 'قلوه‌ای', 'description': 'فرم قلب با برجستگی مرکزی'},
            {'id': 'russian', 'name': 'روسی', 'description': 'لب پُر و عمودی'},
            {'id': 'brazilian', 'name': 'برزیلی', 'description': 'حجم بالا با گوشه‌های بالا'},
            {'id': 'hollywood', 'name': 'هالیوودی', 'description': 'پُر و برجسته'},
            {'id': 'classic', 'name': 'کلاسیک', 'description': 'فرم متعادل و طبیعی'},
            {'id': 'natural', 'name': 'طبیعی', 'description': 'حجم متعادل و روزمره'}
        ],
        'nose': [
            {'id': 'slim_bridge', 'name': 'قلمی', 'description': 'پل بینی باریک و ظریف'},
            {'id': 'doll_tip', 'name': 'عروسکی', 'description': 'نوک بینی گرد و زیبا'},
            {'id': 'natural', 'name': 'طبیعی', 'description': 'فرم متعادل و طبیعی'}
        ],
        'jaw': [
            {'id': 'sharper', 'name': 'تیز', 'description': 'خط فک تیز و مشخص'},
            {'id': 'rounder', 'name': 'گرد', 'description': 'خط فک گرد و نرم'}
        ],
        'cheek': [
            {'id': 'enhance', 'name': 'برجسته', 'description': 'گونه‌های برجسته و مشخص'},
            {'id': 'reduce', 'name': 'طبیعی', 'description': 'گونه‌های طبیعی و متعادل'}
        ],
        'forehead': [
            {'id': 'smooth', 'name': 'صاف', 'description': 'پیشانی صاف و یکدست'},
            {'id': 'enhance', 'name': 'برجسته', 'description': 'پیشانی برجسته و مشخص'}
        ]
    }
    available = styles.get(area, [])
    return {"status": "success", "area": area, "styles": available, "count": len(available)}


@router.get("/edit/actions")
async def get_actions():
    return {
        "status": "success",
        "actions": [
            {"id": "smaller", "name": "کوچک‌تر", "description": "کاهش اندازه و حجم"},
            {"id": "bigger", "name": "بزرگ‌تر", "description": "افزایش اندازه و حجم"},
            {"id": "fuller", "name": "پرتر", "description": "افزایش حجم و برجستگی"},
            {"id": "sharper", "name": "تیزتر", "description": "تیز کردن خطوط و زوایا"},
            {"id": "smoother", "name": "صاف‌تر", "description": "صاف کردن و یکدست‌سازی"},
            {"id": "lift", "name": "لیفت", "description": "بالا بردن و لیفت کردن"}
        ]
    }


@router.get("/edit/health")
async def edit_health():
    return {
        "status": "healthy",
        "model_loaded": beauty_engine is not None,
        "timestamp": time.time()
    }