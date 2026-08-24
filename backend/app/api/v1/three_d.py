from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

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

        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return JSONResponse({"error": "Invalid image"}, status_code=400)

        result = beauty_engine.process(image, text, intensity)

        # When the NLP parser can't map the text to an area/action we still
        # return the UNMODIFIED photo as the "scan" result. Only hard
        # failures (e.g. no face found) are surfaced as 400.
        changes = {
            'area': None,
            'action': None,
            'intensity': intensity or 0,
        }
        source_image = image
        description = 'چهره اسکن شد. برای اعمال تغییر، ناحیه و نوع تغییر را دقیق‌تر توضیح بده.'

        if result.get('status') == 'success':
            source_image = result['image']
            changes = result['changes']
            description = result['description']
            applied_changes = result.get('applied_changes') or [result['changes']]
            view = result.get('view', 'front')
        else:
            applied_changes = []
            view = 'front'
            if 'متوجه نشدم' not in str(result.get('message', '')):
                return JSONResponse({"error": result.get('message', 'پردازش ناموفق بود')}, status_code=400)

        _, buffer = cv2.imencode('.jpg', source_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # ✅ عکس اصلی هم برای نمایش «قبل/بعد» کوچک می‌شود
        try:
            from app.services.filter_service import FilterService
            orig_small = FilterService._downscale_for_app(image)
        except Exception:
            orig_small = source_image
        _, obuf = cv2.imencode('.jpg', orig_small, [cv2.IMWRITE_JPEG_QUALITY, 90])
        original_b64 = base64.b64encode(obuf).decode('utf-8')

        # 🎯 فیلتر دوبعدی (Snapchat-style) — اعمال ترتیبی همه تغییرات درخواستی
        # 🧊 اگر نیم‌رخ باشد، مستقیم از موتور سه‌بعدی پیش‌نمایش چرخیده می‌سازیم
        filtered_b64 = None
        face_meta = None
        three_d_preview_b64 = None
        if changes.get('area'):
            try:
                from app.services.filter_service import filter_service
                from app.services.face_mesh_robust import robust_face_mesh

                lms, face_meta = robust_face_mesh.detect(source_image)
                if lms:
                    # 🆕 چند-تغییره: هر change روی خروجی قبلی اعمال می‌شود
                    # 🎯 بینی از موتور آناتومیک با Poisson آمده — لایه دوم فقط
                    #    برای نواحی غیربینی اجرا می‌شود تا اثر خنثی/سایه ایجاد نکند
                    current = source_image
                    second_pass = [ch for ch in applied_changes
                                   if ch['area'] != 'nose']
                    for ch in second_pass:
                        out_b64 = filter_service.apply(
                            current, lms,
                            area=ch['area'],
                            action=ch.get('action') or 'fuller',
                            intensity=float(ch.get('intensity') or 0.5),
                        )
                        if out_b64:
                            buf = np.frombuffer(
                                base64.b64decode(out_b64), np.uint8)
                            dec = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                            if dec is not None:
                                if dec.shape[:2] != current.shape[:2]:
                                    dec = cv2.resize(
                                        dec, (current.shape[1], current.shape[0]),
                                        interpolation=cv2.INTER_AREA)
                                current = dec
                    if applied_changes:
                        _, fbuf = cv2.imencode(
                            '.jpg', current, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        filtered_b64 = base64.b64encode(fbuf).decode('utf-8')

                    # 🧊 نمای نیم‌رخ → پیش‌نمایش سه‌بعدی چرخیده هم ضمیمه کن
                    if view in ('left_profile', 'right_profile'):
                        try:
                            from app.services.multi_view_3d import (
                                multi_view_reconstructor as mvr,
                            )
                            pts = np.array([[lm['x'] * source_image.shape[1],
                                             lm['y'] * source_image.shape[0]]
                                            for lm in lms], dtype=np.float32)
                            zs = np.array([lm.get('z', 0.0) for lm in lms],
                                          dtype=np.float32) * source_image.shape[1]
                            faces = mvr._triangulate(
                                pts, source_image.shape[1], source_image.shape[0])
                            preview = mvr._render_preview(pts, zs, faces, size=640)
                            if preview:
                                three_d_preview_b64 = preview
                        except Exception as pe:
                            logger.warning(f"profile 3D preview failed: {pe}")
                else:
                    logger.warning(f"face not locked for filter: {face_meta}")
            except Exception as fe:
                logger.warning(f"2D filter failed (non-fatal): {fe}")

        # 🆕 پیشنهاد درمان + پزشک (فقط وقتی ناحیه مشخصی خواسته شده)
        from app.services.doctor_recommendation import (
            recommend_for_change,
            extract_gel_cc,
        )
        gel_cc = extract_gel_cc(text)
        recommendation = recommend_for_change(
            area=changes.get('area'),
            action=changes.get('action'),
            intensity=float(changes.get('intensity') or 0),
            gel_cc=gel_cc,
        )

        return {
            "status": "success",
            "image": img_base64,
            "original_image": original_b64,
            "filtered_image": filtered_b64,
            "three_d_preview": three_d_preview_b64,
            "changes": changes,
            "applied_changes": applied_changes,
            "view": view,
            "description": description,
            "intensity": changes.get('intensity') or 0,
            "recommendation": recommendation or None,
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