"""
🎯 FaceEditEngine v1.0 — موتور حرفه‌ای ادیت چهره
================================================
هر درخواست کاربر → چند استایل مختلف → کاربر انتخاب میکنه
فقط ناحیه مشخص عوض میشه، بقیه صورت ۱۰۰٪ ثابت میمونه.

tapahiha otagh: RTX 570 12GB
pip install: diffusers transformers accelerate safetensors insightface
             opencv-python-headless pillow mediapipe
"""
import os, sys, cv2, numpy as np, torch, base64, json
from PIL import Image
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# ============================================
#   ریشه باگ: مدل الان چیکار میکنه
# ============================================
#
#  ❌ BUG CURRENT: خروجی خام SDXL روی کل عکس اعمال میشه
#     → ماسک غیردقیق، کل صورت عوض میشه
#     → هویت حفظ نمیشه (چشم‌ها/لب‌ها/پیشانی عوض میشه)
#
#  ✅ FIX: پایپلاین ۴ مرحله‌ای با ماسک دقیق + identity guard
#     1. Face Segmentation → ماسک فقط ناحیه هدف
#     2. Identity Extraction → استخراج هویت از بقیه صورت
#     3. Inpaint → فقط داخل ماسک عوض میشه
#     4. Post-process → ترکیب با بافت واقعی پوست

class FaceEditEngine:
    """
    موتور ادیت چهره با کنترل ناحیه دقیق
    
    usage:
        engine = FaceEditEngine(device='cuda:0')
        engine.load()
        
        # ۳ استایل مختلف بساز
        styles = engine.generate_styles(image_bgr, 'nose', 'fantasy', intensity=0.7)
        # → [{'id': 0, 'image': bgr, 'label': 'فانتزی ملایم', 'score': 0.92}, ...]
        
        # کاربر انتخاب کنه
        final = engine.apply_choice(image_bgr, styles[0]['id'])
    """
    
    # ============================================
    #   منطقه‌های قابل ادیت و استایل‌های هرکدام
    # ============================================
    AREA_STYLES = {
        'nose': {
            'fantasy':     ['بینی فانتزی', 'barik-e-pont, ziba va naazok'],
            'slim':        ['بینی قلمی', 'thin straight nose bridge'],
            'doll':        ['بینی عروسکی', 'small cute rounded nose'],
            'upturned':    ['نوک بالا', 'lifted nasal tip'],
            'natural':     ['طبیعی', 'natural refined nose'],
        },
        'lip': {
            'russian':     ['روسی', 'full voluminous russian lips'],
            'brazilian':   ['برزیلی', 'brazilian lips rounded'],
            'hollywood':   ['هالیوودی', 'hollywood full lips'],
            'heart':       ['قلوه‌ای', 'heart-shaped lips'],
            'natural':     ['طبیعی', 'natural lips'],
        },
        'jaw': {
            'sharper':     ['تیز', 'sharp defined jawline'],
            'rounder':     ['گرد', 'soft rounded jaw'],
            'natural':     ['طبیعی', 'natural jawline'],
        },
        'cheek': {
            'enhance':     ['برجسته', 'enhanced cheekbones'],
            'reduce':      ['طبیعی', 'natural cheeks'],
        },
    }
    
    # تعداد استایل‌هایی که برای هر درخواست برمیگردونه
    VARIANTS_PER_STYLE = 3
    
    def __init__(self, device: str = 'cuda:0'):
        self.device = device
        self.pipe = None           # SDXL Inpaint
        self.face_analyzer = None  # InsightFace for identity
        self.loaded = False
    
    def load(self):
        """لود مدل‌ها با cpu_offload برای ۱۲GB"""
        if self.loaded:
            return
        logger.info('Loading FaceEditEngine...')
        from diffusers import StableDiffusionXLInpaintPipeline, AutoPipelineForImage2Image
        from controlnet_aux import MLSDdetector
        
        # SDXL Inpaint — حداکثر کیفیت با fp16
        self.pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            torch_dtype=torch.float16,
            variant='fp16',
            safety_checker=None,
        ).to(self.device)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        
        # InsightFace برای استخراج هویت
        try:
            from insightface.app import FaceAnalysis
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            logger.warning(f'InsightFace unavailable: {e} — identity guard disabled')
        
        self.loaded = True
        logger.info('FaceEditEngine ready')
    
    # ============================================
    #   استخراج ماسک دقیق ناحیه
    # ============================================
    def _make_mask(self, image_bgr: np.ndarray, area: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        ماسک دقیق فقط ناحیه هدف + ایندیکس‌های FaceMesh
        return: (mask_h, mask_w, points_478)
        """
        import mediapipe as mp
        h, w = image_bgr.shape[:2]
        mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            refine_landmarks=True,
            max_num_faces=1,
        )
        res = mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mesh.close()
        
        if not res.multi_face_landmarks:
            raise ValueError('No face detected')
        
        lms = res.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)
        
        # ایندیس‌های دقیق هر ناحیه از FaceMesh 478
        AREA_INDICES = {
            'nose': [
                1,2,4,5,6,19,94,98,97,206,327,326,426,
                168,  # radix
                197,195, # mid bridge
                129,358,  # alar
            ],
            'lip': [
                61,146,91,181,84,17,314,405,321,375,
                291,308,324,318,402,317,14,87,178,88,
                179,89,96,185,40,39,37,0,267,269,270,
                409,291,270,409,
            ],
            'jaw': [
                172,136,150,176,149,177,152,377,400,378,379,365,
                397,288,361,323,454,356,389,251,284,332,
            ],
            'cheek': [
                36,205,206,207,21,50,187,123,50,101,48,115,131,198,209,
                263,425,426,427,280,355,424,398,362,384,385,258,321,311,352,
            ],
        }
        
        indices = AREA_INDICES.get(area, AREA_INDICES['nose'])
        valid = [i for i in indices if i < len(pts)]
        poly = pts[valid].astype(np.int32)
        
        # ماسک دقیق با بلور نرم لبه
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        mask = cv2.dilate(mask, np.ones((3,3), np.uint8), iterations=1)
        mask = cv2.GaussianBlur(mask, (7,7), 2)
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        
        return mask, pts
    
    # ============================================
    #   استخراج هویت چهره
    # ============================================
    def _get_identity(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Embedding هویت از ناحیه غیربینی (چشم‌ها/پیشانی)"""
        if self.face_analyzer is None:
            return None
        try:
            faces = self.face_analyzer.get(image_bgr[:, :, ::-1])  # BGR→RGB
            if faces:
                return faces[0].embedding
        except:
            pass
        return None
    
    def _identity_distance(self, emb1, emb2) -> float:
        """فاصله کسینوس بین دو هویت — کمتر = شبیه‌تر"""
        if emb1 is None or emb2 is None:
            return 0.0
        return float(1 - np.dot(emb1, emb2) / 
                     (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9))
    
    # ============================================
    #   ساخت پرامپت دقیق
    # ============================================
    def _build_prompt(self, area: str, style: str, intensity: float) -> Tuple[str, str]:
        """
        پرامپت مثبت و منفی
        کلید: پرامپت منفی باید دقیقاً بگه چه چیزی عوض نشود
        """
        style_info = self.AREA_STYLES.get(area, {}).get(style, ['refined', 'refined'])
        pos_label, pos_en = style_info
        
        # ناحیه‌هایی که باید حفظ شوند
        protect = {
            'nose':  'eyes, lips, forehead, chin, jaw, ears, hair, skin texture, identity',
            'lip':   'eyes, nose, forehead, chin, jaw, ears, hair, skin texture, identity',
            'jaw':   'eyes, nose, lips, forehead, ears, hair, skin texture, identity',
            'cheek': 'eyes, nose, lips, forehead, chin, jaw, ears, hair, skin texture, identity',
        }
        
        positive = (
            f'RAW photo of same person, {pos_en}, '
            f'photorealistic skin texture, '
            f'exact same identity as original, '
            f'professional beauty photography, '
            f'high resolution, detailed skin pores'
        )
        
        negative = (
            f'cartoon, painting, illustration, drawing, anime, '
            f'deformed {area}, distorted face, '
            f'changed {protect.get(area, "identity")}, '
            f'plastic look, artificial skin, '
            f'artifacts, seams, rings, halos, '
            f'ugly, low quality, blurry'
        )
        
        return positive, negative
    
    # ============================================
    #   تولید چند استایل مختلف
    # ============================================
    def generate_styles(
        self,
        image_bgr: np.ndarray,
        area: str,
        style: str,
        intensity: float = 0.7,
        n_variants: int = 3,
    ) -> List[Dict]:
        """
        چند استایل مختلف بساز تا کاربر انتخاب کنه
        return: [{'id': 0, 'image': bgr, 'label': str, 'score': float}, ...]
        """
        if not self.loaded:
            self.load()
        
        mask, pts = self._make_mask(image_bgr, area)
        identity_orig = self._get_identity(image_bgr)
        
        positive, negative = self._build_prompt(area, style, intensity)
        
        # تغییرات جزئی در strength برای تنوع
        strengths = [0.25, 0.35, 0.45]  # ملایم، متوسط، قوی
        labels = ['ملایم', 'متوسط', 'قوی']
        
        variants = []
        pil_orig = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        pil_mask = Image.fromarray(mask)
        
        for i, (strg, lbl) in enumerate(zip(strengths, labels)):
            try:
                # ترکیب intensity و strength
                eff_strg = min(strg * intensity * 1.5, 0.6)
                
                result = self.pipe(
                    prompt=positive,
                    negative_prompt=negative,
                    image=pil_orig,
                    mask_image=pil_mask,
                    width=pil_orig.width,
                    height=pil_orig.height,
                    strength=eff_strg,
                    guidance_scale=8.0,
                    num_inference_steps=30,
                ).images[0]
                
                result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                
                # ── حفظ هویت: اگر افتاده، تلاش مجدد با strength کمتر ──
                identity_gen = self._get_identity(result_bgr)
                id_distance = self._identity_distance(identity_orig, identity_gen)
                
                if id_distance > 0.15:
                    logger.info(f'variant {i}: identity drift {id_distance:.3f} — retry softer')
                    result = self.pipe(
                        prompt=positive,
                        negative_prompt=negative,
                        image=pil_orig,
                        mask_image=pil_mask,
                        width=pil_orig.width,
                        height=pil_orig.height,
                        strength=eff_strg * 0.5,
                        guidance_scale=8.0,
                        num_inference_steps=30,
                    ).images[0]
                    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
                    identity_gen = self._get_identity(result_bgr)
                    id_distance = self._identity_distance(identity_orig, identity_gen)
                
                # ── کامپوزیت نهایی: فقط داخل ماسک عوض شود ──
                final = self._composite(image_bgr, result_bgr, mask)
                
                # ── امتیاز: شبیه هویت + متفاوت از اصل ──
                diff_score = cv2.absdiff(image_bgr, final).mean() / 255.0
                score = (1 - id_distance) * 0.6 + min(diff_score * 3, 0.4)
                
                variants.append({
                    'id': i,
                    'image': final,
                    'label': f'{self.AREA_STYLES[area][style][0]} — {lbl}',
                    'score': round(score, 3),
                    'identity_distance': round(id_distance, 4),
                    'strength': round(eff_strg, 3),
                })
                
            except Exception as e:
                logger.error(f'variant {i} failed: {e}')
        
        return sorted(variants, key=lambda v: -v['score'])
    
    # ============================================
    #   ترکیب نهایی: فقط داخل ماسک عوض شود
    # ============================================
    def _composite(self, original: np.ndarray, generated: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        تضمین ریاضی: بیرون ماسک = عکس اصلی (۱۰۰٪)
        داخل ماسک = ترکیب نرم با بافت واقعی پوست
        """
        m = mask.astype(np.float32) / 255.0
        
        # فدر نرم لبه ماسک
        m3 = m[..., None]
        
        # تطبیق روشنایی: جلوگیری از تفاوت نور
        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        gen_lab = cv2.cvtColor(generated, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        m_bool = m > 0.01
        if m_bool.sum() > 50:
            L_o = orig_lab[..., 0]; L_g = gen_lab[..., 0]
            mean_o = float(L_o[m_bool].mean())
            mean_g = float(L_g[m_bool].mean())
            # فقط روشنایی را تطبیق بده، رنگ حفظ شود
            gen_lab[..., 0] += (mean_o - mean_g) * m3[..., 0]
            generated = cv2.cvtColor(gen_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # ترکیب: فرم جدید + بافت واقعی پوست
        # High-frequency از اصل، low-frequency از تولیدی
        orig_low = cv2.GaussianBlur(original, (0,0), 10).astype(np.float32)
        gen_low = cv2.GaussianBlur(generated, (0,0), 10).astype(np.float32)
        skin_texture = original.astype(np.float32) - orig_low
        blended = gen_low + skin_texture * 0.8  # بافت واقعی پوست حفظ شود
        
        result = (blended * m3 + original.astype(np.float32) * (1 - m3))
        return np.clip(result, 0, 255).astype(np.uint8)
    
    # ============================================
    #   انتخاب نهایی کاربر
    # ============================================
    def apply_choice(self, image_bgr: np.ndarray, variant_id: int, 
                     variants: List[Dict]) -> Optional[np.ndarray]:
        """استایل انتخابشده توسط کاربر"""
        for v in variants:
            if v['id'] == variant_id:
                return v['image']
        return None
    
    # ============================================
    #   API ساده‌شده
    # ============================================
    def edit(self, image_bgr: np.ndarray, text: str) -> Dict:
        """
        یک متد ساده: عکس + متن → خروجی
        
        اما بهتره از generate_styles استفاده کنی
        تا کاربر بتونه انتخاب کنه
        """
        area, style = self._parse_text(text)
        styles = self.generate_styles(image_bgr, area, style, intensity=0.7)
        
        if not styles:
            return {'status': 'error', 'message': 'تولید ممکن نبود'}
        
        return {
            'status': 'success',
            'styles': styles,
            'best': styles[0],
            'area': area,
            'style': style,
        }
    
    def _parse_text(self, text: str) -> Tuple[str, str]:
        """تشخیص ناحیه و استایل از متن فارسی"""
        text = text.lower()
        
        # تشخیص ناحیه
        area = 'nose'
        if any(k in text for k in ['لب', 'دهان']):
            area = 'lip'
        elif any(k in text for k in ['فک', 'چانه', 'چونه']):
            area = 'jaw'
        elif any(k in text for k in ['گونه']):
            area = 'cheek'
        
        # تشخیص استایل
        style = 'natural'
        if any(k in text for k in ['فانتزی', 'اروپایی', 'fantasy']):
            style = 'fantasy'
        elif any(k in text for k in ['قلمی', 'باریک', 'slim']):
            style = 'slim'
        elif any(k in text for k in ['عروسکی', 'dzoll']):
            style = 'doll'
        elif any(k in text for k in ['بالا', 'sarbala']):
            style = 'upturned'
        elif any(k in text for k in ['روسی']):
            style = 'russian'
        elif any(k in text for k in ['برزیلی']):
            style = 'brazilian'
        elif any(k in text for k in ['هالیوودی', 'holly']):
            style = 'hollywood'
        elif any(k in text for k in ['قلوه', 'قلب']):
            style = 'heart'
        elif any(k in text for k in ['تیز', 'sharp']):
            style = 'sharper'
        elif any(k in text for k in ['گرد', 'round']):
            style = 'rounder'
        elif any(k in text for k in ['برجسته', 'enhance']):
            style = 'enhance'
        elif any(k in text for k in ['کوچک', 'کم', 'reduce', 'natooi']):
            style = 'reduce'
        
        return area, style
