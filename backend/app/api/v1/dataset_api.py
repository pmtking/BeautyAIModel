"""
Dataset & Admin API — ساخت دیتاست‌های قبل/بعد + مدیریت کاربران
=============================================================
- ثبت‌نام / لاگین (توکن ساده)
- ادمین می‌تواند کاربر بسازد
- آپلود دیتاست: عکس قبل + بعد + توضیحات
- لیست / ویرایش / حذف دیتاست
دیتا در فایل JSON ذخیره می‌شود (بدون نیاز به دیتابیس).
"""
import os
import json
import time
import hashlib
import secrets
import shutil
from pathlib import Path
from typing import Optional, List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel

router = APIRouter()

# ============================================================
# تنظیمات
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # backend_full/
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
USERS_FILE = DATA_DIR / "users.json"
DATASETS_FILE = DATA_DIR / "datasets.json"
ADMIN_USERNAME = os.environ.get("DATASET_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("DATASET_ADMIN_PASS", "butiadmin2026")

DATA_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

MAX_IMAGE_SIZE = 15 * 1024 * 1024  # 15MB


# ============================================================
# توابع کمکی فایل JSON
# ============================================================
def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_users() -> dict:
    return load_json(USERS_FILE, {})


def save_users(users: dict):
    save_json(USERS_FILE, users)


def get_datasets() -> list:
    return load_json(DATASETS_FILE, [])


def save_datasets(datasets: list):
    save_json(DATASETS_FILE, datasets)


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def make_token() -> str:
    return secrets.token_hex(24)


# ============================================================
# مدل‌ها
# ============================================================
class RegisterIn(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


class LoginIn(BaseModel):
    username: str
    password: str


class CreateUserIn(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    role: str = "user"


def authenticate(token: Optional[str]) -> dict:
    """توکن → کاربر. اگر admin باشد role=admin"""
    if not token:
        raise HTTPException(status_code=401, detail="توکن الزامی است")
    users = get_users()
    for u in users.values():
        if u.get("token") == token:
            return u
    raise HTTPException(status_code=401, detail="توکن نامعتبر است")


def require_admin(user: dict):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="فقط ادمین اجازه دارد")


# ============================================================
# روت‌های احراز هویت
# ============================================================
@router.post("/dataset/register")
async def register(body: RegisterIn):
    users = get_users()
    if body.username in users:
        raise HTTPException(status_code=400, detail="نام کاربری تکراری است")
    token = make_token()
    users[body.username] = {
        "username": body.username,
        "password": hash_password(body.password),
        "full_name": body.full_name or "",
        "role": "user",
        "token": token,
        "created_at": time.time(),
    }
    save_users(users)
    return {"status": "success", "token": token, "username": body.username, "role": "user"}


@router.post("/dataset/login")
async def login(body: LoginIn):
    users = get_users()
    u = users.get(body.username)

    # 🔑 Bootstrap: ادمین پیش‌فرض (از env یا مقدار پیش‌فرض) همیشه می‌تواند وارد شود
    if body.username == ADMIN_USERNAME and body.password == ADMIN_PASSWORD:
        if u is None:
            u = {
                "username": ADMIN_USERNAME,
                "password": hash_password(ADMIN_PASSWORD),
                "full_name": "Administrator",
                "role": "admin",
                "token": make_token(),
                "created_at": time.time(),
            }
            users[ADMIN_USERNAME] = u
        else:
            u["role"] = "admin"
        u["token"] = make_token()
        save_users(users)
        return {"status": "success", "token": u["token"], "username": u["username"], "role": "admin"}

    if not u or u.get("password") != hash_password(body.password):
        raise HTTPException(status_code=401, detail="نام کاربری یا رمز اشتباه است")
    # توکن جدید
    u["token"] = make_token()
    save_users(users)
    return {"status": "success", "token": u["token"], "username": body.username, "role": u.get("role", "user")}


@router.get("/dataset/me")
async def me(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "") if authorization else None
    u = authenticate(token)
    return {"status": "success", "username": u["username"], "role": u.get("role", "user"), "full_name": u.get("full_name", "")}


# ============================================================
# روت‌های ادمین — ساخت کاربر
# ============================================================
@router.post("/dataset/admin/create-user")
async def admin_create_user(body: CreateUserIn, authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "") if authorization else None
    admin = authenticate(token)
    require_admin(admin)
    users = get_users()
    if body.username in users:
        raise HTTPException(status_code=400, detail="نام کاربری تکراری است")
    token_new = make_token()
    users[body.username] = {
        "username": body.username,
        "password": hash_password(body.password),
        "full_name": body.full_name or "",
        "role": body.role if body.role in ("admin", "user") else "user",
        "token": token_new,
        "created_at": time.time(),
    }
    save_users(users)
    return {"status": "success", "username": body.username, "role": body.role}


@router.get("/dataset/admin/users")
async def admin_list_users(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "") if authorization else None
    admin = authenticate(token)
    require_admin(admin)
    users = get_users()
    result = [
        {"username": u["username"], "role": u.get("role", "user"),
         "full_name": u.get("full_name", ""), "created_at": u.get("created_at")}
        for u in users.values()
    ]
    return {"status": "success", "users": result}


# ============================================================
# روت‌های دیتاست — آپلود / لیست / حذف / ویرایش
# ============================================================
def _save_image(upload: UploadFile, ds_id: str, prefix: str) -> str:
    content = upload.file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=400, detail="حجم تصویر بیشتر از ۱۵MB است")
    ext = os.path.splitext(upload.filename or "")[1] or ".jpg"
    if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
        ext = ".jpg"
    fname = f"{ds_id}_{prefix}{ext}"
    fpath = DATASETS_DIR / fname
    with open(fpath, "wb") as f:
        f.write(content)
    return f"/data/datasets/{fname}"


@router.post("/dataset/upload")
async def upload_dataset(
    before_front: UploadFile = File(...),
    after_front: UploadFile = File(...),
    before_right: Optional[UploadFile] = File(None),
    after_right: Optional[UploadFile] = File(None),
    before_left: Optional[UploadFile] = File(None),
    after_left: Optional[UploadFile] = File(None),
    description: str = Form(""),
    procedure_type: str = Form(""),
    age: str = Form(""),
    gender: str = Form(""),
    injection_amount: str = Form(""),
    area: str = Form(""),
    tags: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "") if authorization else None
    user = authenticate(token)

    ds_id = f"ds_{int(time.time() * 1000)}"

    # عکس‌های اصلی (اجباری)
    before_front_url = _save_image(before_front, ds_id, "before_front")
    after_front_url = _save_image(after_front, ds_id, "after_front")

    # عکس‌های اختیاری
    before_right_url = _save_image(before_right, ds_id, "before_right") if before_right else None
    after_right_url = _save_image(after_right, ds_id, "after_right") if after_right else None
    before_left_url = _save_image(before_left, ds_id, "before_left") if before_left else None
    after_left_url = _save_image(after_left, ds_id, "after_left") if after_left else None

    datasets = get_datasets()
    datasets.append({
        "id": ds_id,
        "before_front_url": before_front_url,
        "after_front_url": after_front_url,
        "before_right_url": before_right_url,
        "after_right_url": after_right_url,
        "before_left_url": before_left_url,
        "after_left_url": after_left_url,
        "description": description,
        "procedure_type": procedure_type,
        "age": age,
        "gender": gender,
        "injection_amount": injection_amount,
        "area": area,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "username": user["username"],
        "created_at": time.time(),
        "updated_at": time.time(),
    })
    save_datasets(datasets)
    return {"status": "success", "id": ds_id, "message": "دیتاست با موفقیت آپلود شد"}


@router.get("/dataset/list")
async def list_datasets(authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "") if authorization else None
    user = authenticate(token)
    datasets = get_datasets()
    # همه می‌توانند ببینند
    result = []
    for ds in datasets:
        result.append({
            "id": ds["id"],
            "before_front_url": ds.get("before_front_url"),
            "after_front_url": ds.get("after_front_url"),
            "before_right_url": ds.get("before_right_url"),
            "after_right_url": ds.get("after_right_url"),
            "before_left_url": ds.get("before_left_url"),
            "after_left_url": ds.get("after_left_url"),
            "description": ds.get("description", ""),
            "procedure_type": ds.get("procedure_type", ""),
            "age": ds.get("age", ""),
            "gender": ds.get("gender", ""),
            "injection_amount": ds.get("injection_amount", ""),
            "area": ds.get("area", ""),
            "tags": ds.get("tags", []),
            "username": ds.get("username", ""),
            "created_at": ds.get("created_at"),
            "updated_at": ds.get("updated_at"),
        })
    return {"status": "success", "datasets": result}


@router.delete("/dataset/{ds_id}")
async def delete_dataset(ds_id: str, authorization: Optional[str] = Header(None)):
    token = authorization.replace("Bearer ", "") if authorization else None
    user = authenticate(token)
    datasets = get_datasets()
    for i, ds in enumerate(datasets):
        if ds["id"] == ds_id:
            # فقط صاحب یا ادمین
            if ds["username"] != user["username"] and user.get("role") != "admin":
                raise HTTPException(status_code=403, detail="فقط صاحب دیتاست یا ادمین می‌تواند حذف کند")
            # حذف فایل‌ها (۶ نمای ممکن)
            img_keys = [
                "before_front_url", "after_front_url",
                "before_right_url", "after_right_url",
                "before_left_url", "after_left_url",
            ]
            for key in img_keys:
                url = ds.get(key)
                if url and url.startswith("/data/datasets/"):
                    p = BASE_DIR / url.lstrip("/")
                    if p.exists():
                        p.unlink()
            datasets.pop(i)
            save_datasets(datasets)
            return {"status": "success", "message": "دیتاست حذف شد"}
    raise HTTPException(status_code=404, detail="دیتاست پیدا نشد")


@router.put("/dataset/{ds_id}")
async def update_dataset(
    ds_id: str,
    description: Optional[str] = Form(None),
    procedure_type: Optional[str] = Form(None),
    age: Optional[str] = Form(None),
    gender: Optional[str] = Form(None),
    injection_amount: Optional[str] = Form(None),
    area: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    before_front: Optional[UploadFile] = File(None),
    after_front: Optional[UploadFile] = File(None),
    before_right: Optional[UploadFile] = File(None),
    after_right: Optional[UploadFile] = File(None),
    before_left: Optional[UploadFile] = File(None),
    after_left: Optional[UploadFile] = File(None),
    authorization: Optional[str] = Header(None),
):
    token = authorization.replace("Bearer ", "") if authorization else None
    user = authenticate(token)
    datasets = get_datasets()
    for ds in datasets:
        if ds["id"] == ds_id:
            if ds["username"] != user["username"] and user.get("role") != "admin":
                raise HTTPException(status_code=403, detail="فقط صاحب دیتاست یا ادمین می‌تواند ویرایش کند")
            # فیلدهای متنی
            field_map = {
                "description": description,
                "procedure_type": procedure_type,
                "age": age,
                "gender": gender,
                "injection_amount": injection_amount,
                "area": area,
            }
            for k, v in field_map.items():
                if v is not None:
                    ds[k] = v
            if tags is not None:
                ds["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
            # عکس‌ها (۶ نما)
            img_fields = {
                "before_front": before_front, "after_front": after_front,
                "before_right": before_right, "after_right": after_right,
                "before_left": before_left, "after_left": after_left,
            }
            for prefix, upload in img_fields.items():
                if upload is not None:
                    key = f"{prefix}_url"
                    old = ds.get(key)
                    if old and old.startswith("/data/datasets/"):
                        p = BASE_DIR / old.lstrip("/")
                        if p.exists():
                            p.unlink()
                    ds[key] = _save_image(upload, ds_id, prefix)
            ds["updated_at"] = time.time()
            save_datasets(datasets)
            return {"status": "success", "message": "دیتاست ویرایش شد"}
    raise HTTPException(status_code=404, detail="دیتاست پیدا نشد")


# ============================================================
# (سرو تصاویر از طریق mount /data در main.py انجام می‌شود)
# ============================================================