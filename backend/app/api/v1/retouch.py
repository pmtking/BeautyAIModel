from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/retouch")
async def retouch_image(
    file: UploadFile = File(...),
    adjustments: str = Form('{}'),  # JSON: {"brighten": 40, "smooth": 60, ...}
):
    """
    🎨 ابزارهای ادیت حرفه‌ای (Peachy-style)

    adjustments keys (0..100 each):
      brighten   — روشن‌کردن سایه‌های پوست
      even_skin  — یکنواخت‌سازی رنگ پوست / کاهش لکه
      smooth     — لطافت پوست با حفظ جزئیات
      warmth     — گرمی رنگ (50=خنثی، >50 گرم، <50 سرد)
      clarity    — شفافیت / کنتراست محلی
      contrast   — کنتراست کلی
      saturation — اشباع رنگ
      vignette   — وینت نرم
    """
    try:
        from app.services.retouch_service import retouch_service

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        try:
            adj = json.loads(adjustments or '{}')
        except json.JSONDecodeError:
            adj = {}

        # skin mask از MediaPipe (اگر در دسترس بود) برای رتوش دقیق‌تر
        landmarks = None
        try:
            from app.services.face_mesh_robust import robust_face_mesh
            landmarks, _meta = robust_face_mesh.detect(image)
        except Exception as me:
            logger.warning(f"landmark detection skipped: {me}")

        result_b64 = retouch_service.apply_all(image, landmarks, adj)
        if result_b64 is None:
            return JSONResponse({"error": "retouch failed"}, status_code=500)

        return {
            "status": "success",
            "image": result_b64,
            "applied": adj,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/retouch/presets")
async def get_presets():
    """پریست‌های آماده برای UI."""
    return {
        "presets": [
            {"id": "natural",  "name": "طبیعی",    "emoji": "🌿", "adjustments": {"smooth": 35, "brighten": 20, "even_skin": 25}},
            {"id": "glow",     "name": "درخشش",    "emoji": "✨", "adjustments": {"brighten": 45, "smooth": 55, "warmth": 65, "even_skin": 30}},
            {"id": "porcelain","name": "مرواریدی", "emoji": "🤍", "adjustments": {"smooth": 70, "brighten": 50, "even_skin": 45, "saturation": 42}},
            {"id": "golden",   "name": "طلایی",    "emoji": "🌅", "adjustments": {"warmth": 80, "brightness": 30, "clarity": 60, "saturation": 58}},
            {"id": "cool",     "name": "خنک",      "emoji": "❄️", "adjustments": {"warmth": 25, "clarity": 55, "contrast": 58}},
            {"id": "sharp",    "name": "وضوح بالا", "emoji": "💎", "adjustments": {"clarity": 75, "contrast": 62, "sharpness": 0}},
        ],
        "tools": [
            {"key": "brighten",   "name": "روشنایی",   "icon": "sun"},
            {"key": "even_skin",  "name": "یکنواختی",  "icon": "droplet"},
            {"key": "smooth",     "name": "لطافت",     "icon": "feather"},
            {"key": "warmth",     "name": "گرمی",      "icon": "thermometer"},
            {"key": "clarity",    "name": "شفافیت",    "icon": "aperture"},
            {"key": "contrast",   "name": "کنتراست",   "icon": "circle-half"},
            {"key": "saturation", "name": "اشباع",     "icon": "palette"},
            {"key": "vignette",   "name": "وینت",      "icon": "focus"},
        ],
    }
