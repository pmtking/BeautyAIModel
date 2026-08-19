# app/services/image_editor.py
# pmtking @copyright 2026 all rights reserved mohammad taheri

import cv2 
import numpy as np
import base64 
from typing import Dict , Any 


class ImageEditor : 
    
    
    def __init__(self) :
        pass 
    def aply_minor_edit(self , image_bytes:bytes ,  edit_type:str = 'nose_smaller') -> Dict[str , Any] :
            """
        اعمال تغییر ساده روی تصویر و بازگشت به صورت Base64
        
        Args:
            image_bytes: بایت‌های تصویر ورودی
            edit_type: نوع تغییر (nose_smaller, nose_bigger, lips_full)
            
        Returns:
            dict: شامل وضعیت، تصویر Base64 و اطلاعات
        """