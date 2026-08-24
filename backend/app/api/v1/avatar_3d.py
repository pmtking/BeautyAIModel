from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/avatar-3d")
async def build_avatar_3d(
    files: list[UploadFile] = File(..., description="۱ تا ۴ عکس — اولی = نمای جلو"),
):
    """
    🧊 بازسازی سه‌بعدی دقیق صورت از چند زاویه.

    ورودی:
      - عکس جلو (اجباری، فایل اول)
      - نیم‌رخ چپ/راست (اختیاری — جزئیات پروفایل بینی/چانه را اضافه می‌کند)

    فرایند:
      1. FaceMesh 478 نقطه‌ای روی هر عکس (با retry های robust)
      2. تخمین yaw هر نما → هم‌ترازی Similarity به قاب نمای جلو
      3. فیوژن عمق وزنی + تقویت سیلوئت پروفایل از نیم‌رخ
      4. مثلث‌بندی Delaunay + بافت از خود عکس
      5. پیش‌نمایش رندر چرخیده

    خروجی:
      mesh (vertices/uvs/faces) + texture + preview (base64)
    """
    try:
        from app.services.multi_view_3d import multi_view_reconstructor

        images = []
        for f in files[:4]:
            content = await f.read()
            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                # محدود کردن اندازه برای سرعت (کیفیت کافی)
                h, w = img.shape[:2]
                if max(h, w) > 1600:
                    s = 1600.0 / max(h, w)
                    img = cv2.resize(img, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_AREA)
                images.append(img)

        if not images:
            return JSONResponse({"error": "هیچ تصویر معتبری ارسال نشد"}, status_code=400)

        result = multi_view_reconstructor.reconstruct(images)
        if not result.get('ok'):
            return JSONResponse({"error": result.get('error', 'بازسازی ناموفق')},
                                status_code=400)

        return {
            "status": "success",
            "views_used": result['views_used'],
            "yaws": result['yaws'],
            "mesh": result['mesh'],
            "texture": result['texture'],
            "preview": result['preview'],
            "message": f"آواتار سه‌بعدی از {result['views_used']} نمای ساخته شد",
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/avatar-3d/test")
async def test_avatar():
    return {"status": "ok", "endpoint": "/api/v1/avatar-3d (POST, multipart files[])"}
