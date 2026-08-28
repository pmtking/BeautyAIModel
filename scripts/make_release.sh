#!/bin/bash
# ساخت پکیج دانلودی برای BUTI Backend
# خروجی: releases/buti-vYYYYMMDD.zip
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER="$(date +%Y%m%d-%H%M)"
OUT="$ROOT/releases"
mkdir -p "$OUT"
NAME="buti-v$VER"
TMP="/tmp/$NAME"
rm -rf "$TMP" && mkdir -p "$TMP"

# فایل‌های ضروری (بدون venv، بدون دیتا، بدون node_modules)
mkdir -p "$TMP/backend/app/static"
mkdir -p "$TMP/backend/app/api/v1"
mkdir -p "$TMP/backend/app/services"
mkdir -p "$TMP/backend/app/core"
mkdir -p "$TMP/backend/app/data/datasets"
mkdir -p "$TMP/docs"
mkdir -p "$TMP/scripts"

cp backend/app/main.py                                "$TMP/backend/app/"
cp backend/app/__init__.py                            "$TMP/backend/app/"
cp backend/app/api/v1/*.py                            "$TMP/backend/app/api/v1/"
cp backend/app/api/v1/__init__.py                     "$TMP/backend/app/api/v1/"
cp backend/app/services/chat_bot.py                   "$TMP/backend/app/services/"
cp backend/app/services/generative.py                 "$TMP/backend/app/services/"
cp backend/app/services/face_detector.py              "$TMP/backend/app/services/"
cp backend/app/services/face_mesh_robust.py           "$TMP/backend/app/services/"
cp backend/app/services/three_d_face_service.py       "$TMP/backend/app/services/"
cp backend/app/services/*.py                          "$TMP/backend/app/services/" 2>/dev/null || true
cp backend/app/static/index.html                      "$TMP/backend/app/static/"
cp backend/app/static/live_mirror.html                "$TMP/backend/app/static/"
cp backend/requirements.txt                           "$TMP/backend/"
cp backend/run.py                                     "$TMP/backend/" 2>/dev/null || true
cp INSTALL.md                                         "$TMP/"
cp docs/PROJECT_BRIEF.md                              "$TMP/docs/"
cp scripts/*.sh                                       "$TMP/scripts/" 2>/dev/null || true

# placeholder برای data
echo "{}" > "$TMP/backend/app/data/users.json"
echo "[]" > "$TMP/backend/app/data/datasets.json"
touch "$TMP/backend/app/data/datasets/.gitkeep"

cd /tmp
rm -f "$OUT/$NAME.zip"
zip -r "$OUT/$NAME.zip" "$NAME" -x "*/__pycache__/*" "*.pyc" "*/node_modules/*" > /dev/null
echo "✅ ساخته شد: $OUT/$NAME.zip"
ls -lh "$OUT/$NAME.zip"
