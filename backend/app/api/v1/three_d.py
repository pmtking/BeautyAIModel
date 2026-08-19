from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import sys
from pathlib import Path

# اضافه کردن مسیر مدل
MODEL_PATH = Path(__file__).parent.parent.parent.parent.parent / 'ai_training/src/models'
sys.path.insert(0, str(MODEL_PATH))

router = APIRouter()


@router.post("/3d-filter")
async def create_3d_filter(
    file: UploadFile = File(...),
    text: str = Form(...),
    intensity: float = Form(0.5)
):
    """
    🎨 تبدیل تصویر به فیلتر سه‌بعدی با اعمال تغییرات زیبایی
    """
    try:
        from beauty_engine.model import beauty_engine
        from app.services.three_d_face_service import three_d_service

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        result = beauty_engine.process(image, text, intensity)

        if result['status'] == 'error':
            return JSONResponse({"error": result['message']}, status_code=400)

        three_d_data = three_d_service.create_3d_texture(
            result['image'],
            {
                'area': result['changes'].get('area'),
                'action': result['changes'].get('action'),
                'intensity': result['intensity']
            }
        )

        if 'error' in three_d_data:
            return JSONResponse({"error": three_d_data['error']}, status_code=400)

        _, buffer = cv2.imencode('.jpg', result['image'], [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "status": "success",
            "image": img_base64,
            "three_d": three_d_data,
            "changes": result['changes'],
            "description": result['description'],
            "intensity": result['intensity']
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/3d-test")
async def test_3d():
    return {
        "status": "ok",
        "message": "3D service is ready",
        "endpoint": "/api/v1/3d-filter (POST)"
    }