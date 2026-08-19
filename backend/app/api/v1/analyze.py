# app/api/v1/analyze.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.face_service import FaceService

router = APIRouter()
face_service = FaceService()


@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    تحلیل تصویر و استخراج نقاط کلیدی صورت
    
    - **file**: تصویر (JPG/PNG)
    """
    try:
        image_bytes = await file.read()
        result = face_service.analyze_image(image_bytes)
        
        if result['status'] == 'error':
            raise HTTPException(status_code=400, detail=result['message'])
        
        return {
            "status": "success",
            "landmarks": result['landmarks'],
            "count": result['count'],
            "features": result['features']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/test")
async def test_analyze():
    """تست وضعیت سرویس تحلیل"""
    return {
        "status": "ok",
        "message": "Analysis service is ready"
    }