# backend/app/core/algorithm.py
"""
الگوریتم اصلی BeautyAI - هسته مرکزی سیستم
"""

import cv2
import numpy as np
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class AlgorithmStep(Enum):
    """مراحل الگوریتم"""
    INPUT = "input"
    PREPROCESS = "preprocess"
    FEATURE_EXTRACTION = "feature_extraction"
    NLP_ANALYSIS = "nlp_analysis"
    WARPING = "warping"
    BLENDING = "blending"
    OUTPUT = "output"


@dataclass
class ProcessingContext:
    """زمینه پردازش - داده‌های بین مراحل"""
    image: Optional[np.ndarray] = None
    landmarks: Optional[List[Dict]] = None
    parsed_request: Optional[Dict] = None
    warped_image: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    result_image: Optional[np.ndarray] = None
    processing_time: float = 0.0
    step_results: Dict[str, Any] = None

    def __post_init__(self):
        self.step_results = {}


class BeautyAlgorithm:
    """
    الگوریتم اصلی BeautyAI
    مدیریت کامل فرآیند از ورودی تا خروجی
    """

    def __init__(self):
        self.steps = []
        self._init_steps()

    def _init_steps(self):
        """ثبت مراحل الگوریتم"""
        self.steps = [
            self.step_input,
            self.step_preprocess,
            self.step_feature_extraction,
            self.step_nlp_analysis,
            self.step_warping,
            self.step_blending,
            self.step_output
        ]

    def process(self, image_bytes: bytes, text: str) -> Dict[str, Any]:
        """
        اجرای کامل الگوریتم

        Args:
            image_bytes: بایت‌های تصویر
            text: متن درخواست کاربر

        Returns:
            dict: نتیجه نهایی
        """
        start_time = time.time()
        context = ProcessingContext()

        try:
            # مرحله ۱: دریافت ورودی
            context = self.step_input(image_bytes, text, context)

            # مرحله ۲: پیش‌پردازش
            context = self.step_preprocess(context)

            # مرحله ۳: استخراج ویژگی‌ها
            context = self.step_feature_extraction(context)

            # مرحله ۴: تحلیل NLP
            context = self.step_nlp_analysis(context)

            # مرحله ۵: Warping
            context = self.step_warping(context)

            # مرحله ۶: Blending
            context = self.step_blending(context)

            # مرحله ۷: خروجی
            context = self.step_output(context)

            # ثبت زمان کل
            context.processing_time = time.time() - start_time

            # تولید گزارش
            return self._generate_report(context)

        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'step': context.step_results.get('current_step', 'unknown'),
                'processing_time': time.time() - start_time
            }

    # ============================================
    # مرحله ۱: دریافت ورودی
    # ============================================
    def step_input(self, image_bytes: bytes, text: str, context: ProcessingContext) -> ProcessingContext:
        """دریافت ورودی کاربر"""
        context.step_results['current_step'] = 'input'
        context.step_results['input_text'] = text
        context.step_results['input_size'] = len(image_bytes)

        # خواندن تصویر
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("تصویر معتبر نیست")

        context.image = image
        context.step_results['input_success'] = True

        return context

    # ============================================
    # مرحله ۲: پیش‌پردازش
    # ============================================
    def step_preprocess(self, context: ProcessingContext) -> ProcessingContext:
        """پیش‌پردازش تصویر"""
        context.step_results['current_step'] = 'preprocess'
        image = context.image

        # ۱. تغییر اندازه
        h, w = image.shape[:2]
        if h > 512 or w > 512:
            scale = min(512/h, 512/w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # ۲. نرمال‌سازی
        image = image.astype(np.float32) / 255.0

        # ۳. تشخیص چهره (ساده)
        # در اینجا Haar Cascade برای تشخیص چهره
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        if len(faces) == 0:
            raise ValueError("چهره‌ای شناسایی نشد")

        # ذخیره اطلاعات
        context.step_results['face_bbox'] = faces[0].tolist()
        context.step_results['preprocess_success'] = True

        return context

    # ============================================
    # مرحله ۳: استخراج ویژگی‌ها
    # ============================================
    def step_feature_extraction(self, context: ProcessingContext) -> ProcessingContext:
        """استخراج نقاط کلیدی"""
        context.step_results['current_step'] = 'feature_extraction'

        # اینجا از FaceParser استفاده می‌کنیم
        # برای سادگی، نقاط شبیه‌سازی می‌شوند
        h, w = context.image.shape[:2]

        # شبیه‌سازی ۴۶۸ نقطه
        landmarks = []
        for i in range(468):
            angle = (i / 468) * 2 * np.pi
            radius = 50 + 30 * np.sin(i / 20)
            x = 150 + radius * np.cos(angle)
            y = 150 + radius * np.sin(angle) * 0.8
            landmarks.append({
                'x': x / w * 100,
                'y': y / h * 100,
                'z': 0
            })

        context.landmarks = landmarks
        context.step_results['landmarks_count'] = len(landmarks)
        context.step_results['feature_extraction_success'] = True

        return context

    # ============================================
    # مرحله ۴: تحلیل NLP
    # ============================================
    def step_nlp_analysis(self, context: ProcessingContext) -> ProcessingContext:
        """تحلیل درخواست کاربر"""
        context.step_results['current_step'] = 'nlp_analysis'

        text = context.step_results.get('input_text', '')
        text_lower = text.lower()

        # تشخیص ناحیه
        areas = {
            'nose': ['بینی', 'دماغ', 'nose'],
            'lip': ['لب', 'دهان', 'lip'],
            'jaw': ['فک', 'چانه', 'jaw'],
            'cheek': ['گونه', 'cheek']
        }

        actions = {
            'smaller': ['کوچک‌تر', 'باریک‌تر', 'smaller'],
            'bigger': ['بزرگ‌تر', 'بلندتر', 'bigger'],
            'fuller': ['پرتر', 'حجم‌تر', 'fuller'],
            'sharper': ['تیزتر', 'مشخص‌تر', 'sharper']
        }

        # تشخیص ناحیه
        area = None
        for a, keywords in areas.items():
            if any(kw in text_lower for kw in keywords):
                area = a
                break

        # تشخیص عمل
        action = None
        for a, keywords in actions.items():
            if any(kw in text_lower for kw in keywords):
                action = a
                break

        # تشخیص شدت
        intensity = 0.5
        if 'کمی' in text_lower or 'یکم' in text_lower:
            intensity = 0.3
        elif 'خیلی' in text_lower or 'زیاد' in text_lower:
            intensity = 0.8

        context.parsed_request = {
            'area': area,
            'action': action,
            'intensity': intensity,
            'original_text': text
        }

        context.step_results['nlp_analysis_success'] = True

        return context

    # ============================================
    # مرحله ۵: Warping
    # ============================================
    def step_warping(self, context: ProcessingContext) -> ProcessingContext:
        """اعمال تغییر شکل"""
        context.step_results['current_step'] = 'warping'

        if not context.parsed_request or not context.parsed_request.get('area'):
            context.warped_image = context.image
            return context

        # استخراج اطلاعات
        area = context.parsed_request['area']
        action = context.parsed_request['action']
        intensity = context.parsed_request['intensity']

        # شبیه‌سازی Warping (در واقعیت مدل‌ها را صدا می‌زند)
        image = context.image.copy()
        h, w = image.shape[:2]

        # نقاط فرضی برای ناحیه
        points = []
        if area == 'nose':
            center = (w//2, int(h*0.55))
            for i in range(10):
                angle = (i/10) * 2 * np.pi
                radius = w * 0.08
                px = center[0] + radius * np.cos(angle)
                py = center[1] + radius * np.sin(angle) * 0.7
                points.append([int(px), int(py)])
        elif area == 'lip':
            center = (w//2, int(h*0.65))
            for i in range(10):
                angle = (i/10) * 2 * np.pi
                radius = w * 0.12
                px = center[0] + radius * np.cos(angle)
                py = center[1] + radius * np.sin(angle) * 0.4
                points.append([int(px), int(py)])
        else:
            context.warped_image = image
            return context

        # اعمال Warping (ساده)
        if points:
            pts = np.array(points, dtype=np.float32)
            center = np.mean(pts, axis=0).astype(int)
            scale = 1 - (intensity * 0.3) if action == 'smaller' else 1 + (intensity * 0.3)

            # ایجاد نقشه تغییر
            map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
            map_x = map_x.astype(np.float32)
            map_y = map_y.astype(np.float32)

            max_dist = 0
            for pt in pts:
                dist = np.sqrt((pt[0]-center[0])**2 + (pt[1]-center[1])**2)
                max_dist = max(max_dist, dist)

            if max_dist > 0:
                for i in range(h):
                    for j in range(w):
                        dist = np.sqrt((i-center[1])**2 + (j-center[0])**2)
                        if dist < max_dist:
                            factor = 1 - (dist/max_dist) * (1-scale)
                            new_x = center[0] + (j-center[0]) * factor
                            new_y = center[1] + (i-center[1]) * factor
                            map_x[i, j] = np.clip(new_x, 0, w-1)
                            map_y[i, j] = np.clip(new_y, 0, h-1)

                warped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
                context.warped_image = warped

                # ایجاد ماسک
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
                context.mask = cv2.GaussianBlur(mask, (21, 21), 0)

        context.step_results['warping_success'] = True

        return context

    # ============================================
    # مرحله ۶: Blending
    # ============================================
    def step_blending(self, context: ProcessingContext) -> ProcessingContext:
        """ترکیب تصاویر"""
        context.step_results['current_step'] = 'blending'

        if context.warped_image is None or context.mask is None:
            context.result_image = context.image
            return context

        try:
            # Poisson Blending
            h, w = context.image.shape[:2]
            moments = cv2.moments(context.mask)
            if moments['m00'] != 0:
                cx = int(moments['m10'] / moments['m00'])
                cy = int(moments['m01'] / moments['m00'])
            else:
                cx, cy = w//2, h//2

            result = cv2.seamlessClone(
                context.warped_image,
                context.image,
                context.mask,
                (cx, cy),
                cv2.NORMAL_CLONE
            )
            context.result_image = result

        except Exception as e:
            # Fallback: ترکیب ساده
            mask_norm = context.mask / 255.0
            mask_norm = np.expand_dims(mask_norm, axis=2)
            result = context.image * (1 - mask_norm) + context.warped_image * mask_norm
            context.result_image = result.astype(np.uint8)

        context.step_results['blending_success'] = True

        return context

    # ============================================
    # مرحله ۷: خروجی
    # ============================================
    def step_output(self, context: ProcessingContext) -> ProcessingContext:
        """تولید خروجی نهایی"""
        context.step_results['current_step'] = 'output'

        if context.result_image is None:
            context.result_image = context.image

        context.step_results['output_success'] = True

        return context

    # ============================================
    # گزارش نهایی
    # ============================================
    def _generate_report(self, context: ProcessingContext) -> Dict:
        """تولید گزارش نهایی"""
        return {
            'status': 'success',
            'steps': context.step_results,
            'processing_time': context.processing_time,
            'result': {
                'has_image': context.result_image is not None,
                'shape': context.result_image.shape if context.result_image is not None else None,
                'landmarks_count': len(context.landmarks) if context.landmarks else 0,
                'parsed_request': context.parsed_request
            }
        }


# ============================================
# استفاده از الگوریتم
# ============================================
if __name__ == "__main__":
    # ایجاد نمونه
    algorithm = BeautyAlgorithm()

    # تست با داده‌های نمونه
    with open("test.jpg", "rb") as f:
        image_bytes = f.read()

    result = algorithm.process(
        image_bytes=image_bytes,
        text="دماغم رو کمی کوچیکتر کن"
    )

    print("📊 نتیجه پردازش:")
    print(f"وضعیت: {result['status']}")
    print(f"زمان پردازش: {result['processing_time']:.3f} ثانیه")
    print(f"تعداد نقاط: {result['result']['landmarks_count']}")
    print(f"درخواست: {result['result']['parsed_request']}")