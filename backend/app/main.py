# app/main.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# ایمپورت روت‌ها
from app.api.v1 import analyze, edit, three_d, retouch, manual_edit, avatar_3d

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


@app.get("/")
async def root():
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


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )