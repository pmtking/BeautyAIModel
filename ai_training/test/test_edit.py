# ai_training/test/test_edit.py
import sys
import os
from pathlib import Path

# اضافه کردن مسیر
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))

import cv2
from src.models.beauty_engine.model import beauty_engine

# خواندن تصویر
image = cv2.imread("test_images/test.jpg")

if image is None:
    print("❌ تصویر پیدا نشد")
else:
    # تست ویرایش با درخواست‌های مختلف
    requests = [
        "دماغم رو کمی کوچیکتر کن",
        "لبامو پرتر کن",
    ]
    
    for req in requests:
        print(f"\n📝 درخواست: {req}")
        result = beauty_engine.process(image, req)
        
        if result['status'] == 'success':
            print(f"   ✅ {result['description']}")
            # ذخیره نتیجه
            filename = f"result_{req[:5]}.jpg"
            cv2.imwrite(filename, result['image'])
            print(f"   💾 ذخیره شد: {filename}")
        else:
            print(f"   ❌ {result['message']}")