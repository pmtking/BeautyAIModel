import sys
import os
from pathlib import Path

# اضافه کردن مسیر src
BASE_DIR = Path(__file__).parent.parent  # ai_training
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / 'src'))


import cv2
from src.models.beauty_engine.model import beauty_engine 


image = cv2.imread("test_images/test.jpg") 

if image is  None : 
    print('تصویر  پیدا  نشد ') 
    
else:
    landmarks = beauty_engine.face_parser.detect_from_image(image) 

    if  landmarks : 
        print(f"✅ {len(landmarks)} نقطه کلیدی شناسایی شد") 
        print(f"📌 نمونه: {landmarks[:3]}")
    else:
        print("❌ چهره‌ای شناسایی نشد")