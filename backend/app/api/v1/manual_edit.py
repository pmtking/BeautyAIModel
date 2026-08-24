from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/manual-edit")
async def manual_edit(
    file: UploadFile = File(...),
    edits: str = Form(...),  # JSON: {"lip": +40, "cheek": -20, "nose": -30}
):
    """
    🎛️ ادیت دستی نواحی صورت — مثل اسلایدرهای Facetune/Snapchat

    edits: دیکشنری ناحیه → شدت (-100 تا +100)
      مثبت = پرتر/بزرگ‌تر | منفی = کوچک‌تر/باریک‌تر

    نواحی: lip, nose, jaw, cheek, eye, forehead
    همه تغییرات در یک پاس روی یک عکس اعمال می‌شوند.
    """
    try:
        from app.services.filter_service import (
            filter_service,
            AREA_POLYGONS,
            AREA_LABELS,
        )
        from app.services.face_mesh_robust import robust_face_mesh

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        try:
            edits_map = json.loads(edits or '{}')
        except json.JSONDecodeError:
            return JSONResponse({"error": "Invalid edits JSON"}, status_code=400)

        # فیلتر ورودی‌های معتبر
        clean: dict[str, float] = {}
        for area, val in edits_map.items():
            if area in AREA_POLYGONS and isinstance(val, (int, float)) and abs(val) >= 1:
                clean[area] = max(-100.0, min(100.0, float(val)))

        if not clean:
            return JSONResponse({"error": "هیچ تغییری مشخص نشده"}, status_code=400)

        lms, meta = robust_face_mesh.detect(image)
        if not lms:
            return JSONResponse({"error": "چهره‌ای شناسایی نشد"}, status_code=400)

        # اعمال ترتیبی: هر ناحیه روی خروجی قبلی
        current = image
        applied: dict[str, float] = {}
        for area, val in clean.items():
            action = 'fuller' if val > 0 else 'smaller'
            out_b64 = filter_service.apply(
                current, lms,
                area=area,
                action=action,
                intensity=abs(val) / 100.0,
            )
            if out_b64:
                buf = np.frombuffer(__import__('base64').b64decode(out_b64), np.uint8)
                current = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                applied[area] = val

        _, buffer = cv2.imencode('.jpg', current, [cv2.IMWRITE_JPEG_QUALITY, 92])
        result_b64 = __import__('base64').b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "image": result_b64,
            "applied": applied,
            "labels": {a: AREA_LABELS.get(a, a) for a in applied},
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
