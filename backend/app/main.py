# app/main.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ایمپورت روت‌ها
from app.api.v1 import analyze, edit, three_d, retouch, manual_edit, avatar_3d, chat, dataset_api

app = FastAPI(
    title="BeautyAI API",
    description="API for AI-powered beauty simulation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
app.include_router(edit.router, prefix="/api/v1", tags=["Edit"])
app.include_router(three_d.router, prefix="/api/v1", tags=["3D"])
app.include_router(retouch.router, prefix="/api/v1", tags=["Retouch"])
app.include_router(manual_edit.router, prefix="/api/v1", tags=["ManualEdit"])
app.include_router(avatar_3d.router, prefix="/api/v1", tags=["Avatar3D"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(dataset_api.router, prefix="/api/v1", tags=["Dataset"])

from app.api.v1 import face_edit
app.include_router(face_edit.router, prefix="/api/v1", tags=["FaceEdit"])


@app.get("/")
async def root():
    index = _STATIC / "index.html"
    if index.exists():
        return FileResponse(index, media_type="text/html")
    return {
        "status": "online",
        "message": "BeautyAIModel API is running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0"
    }


# ---------- صفحه آینه هوشمند (دموی زنده روی موبایل) ----------
_STATIC = Path(__file__).resolve().parent / "static"
# دیتاست‌ها در backend/app/data/ ذخیره می‌شوند (BASE_DIR در dataset_api)
_DATA = Path(__file__).resolve().parent / "data"

# سرو تصاویر دیتاست
from fastapi.staticfiles import StaticFiles
if _DATA.exists():
    app.mount("/data", StaticFiles(directory=str(_DATA)), name="data")


@app.get("/mirror")
async def live_mirror():
    return FileResponse(_STATIC / "live_mirror.html", media_type="text/html")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )