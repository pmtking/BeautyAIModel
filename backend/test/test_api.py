# test_api.py
"""
تست کامل API BeautyAI
"""

import requests
import base64
from PIL import Image
from io import BytesIO

BASE_URL = "http://localhost:8000"

def test_health():
    print("🔍 تست سلامت...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"✅ {response.json()}\n")

def test_analyze():
    print("🔍 تست آنالیز صورت...")
    with open("test.jpg", "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        response = requests.post(f"{BASE_URL}/api/v1/analyze", files=files)
    
    data = response.json()
    print(f"✅ وضعیت: {data.get('status')}")
    print(f"📊 تعداد نقاط: {data.get('count')}")
    print(f"⏱️ زمان: {data.get('processing_time')}s\n")

def test_edit():
    print("🔍 تست ویرایش صورت...")
    with open("test.jpg", "rb") as f:
        files = {"file": ("test.jpg", f, "image/jpeg")}
        data = {"text": "دماغم رو کوچیکتر کن", "intensity": "0.5"}
        response = requests.post(f"{BASE_URL}/api/v1/edit", files=files, data=data)
    
    result = response.json()
    print(f"✅ وضعیت: {result.get('status')}")
    print(f"📝 توضیحات: {result.get('data', {}).get('description')}")
    print(f"⏱️ زمان: {result.get('processing_time')}s")
    
    # ذخیره تصویر
    img_data = result.get('data', {}).get('image')
    if img_data:
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        img.save("result_edit.jpg")
        print("💾 تصویر ذخیره شد: result_edit.jpg\n")

def test_styles():
    print("🔍 تست دریافت استایل‌ها...")
    response = requests.get(f"{BASE_URL}/api/v1/edit/styles?area=lip")
    print(f"✅ استایل‌های لب: {response.json().get('styles')}\n")

if __name__ == "__main__":
    test_health()
    test_analyze()
    test_edit()
    test_styles()