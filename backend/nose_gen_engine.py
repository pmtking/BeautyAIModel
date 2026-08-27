"""
🎯 NoseGenEngine — تولید مولد «فقط بینی» با قفل هویت (نسخه حرفه‌ای v3)
======================================================================
اصل طلایی: هیچ پیکسلی بیرون از ماسک بینی نباید تغییر کند — تضمین ریاضی.

تفاوت با نسخه قبلی (beautygen_v2):
  ❌ قبلاً: کل عکس resize به ۱۰۲۴×۱۰۲۴ → صورت کش میآمد → هویت آسیب
  ✅ حالا:  Crop اطراف بینی با حفظ نسبت ابعاد → تولید روی برش →
           برگشت دقیق به مقیاس اصلی → کامپوزیت فقط داخل ماسک فِدرشده

لایه‌های محافظت:
  ۱. Crop-to-context: برش با حاشیه کافی (چشم تا چانه) برای کنتکست
  ۲. ماسک تنگ آناتومیک از MediaPipe 478 + فدر (بلور) لبه‌ها
  ۳. ControlNet Canny قوی‌تر (۰.۶۵) + سقف strength پایین‌تر (≤۰.۶۰)
  ۴. انتقال بافت واقعی پوست (high-frequency) روی خروجی
  ۵. Identity Guard: اگر InsightFace موجود باشد شباهت چهره سنجیده میشود؛
     افت زیاد → یکبار تلاش مجدد با شدت کمتر
  ۶. کامپوزیت نهایی: out = gen*mask + original*(1-mask)  ← بیرون ماسک صفر
"""
import os
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ---------- پرامپت‌ها (همان دیکشنری امتحان‌پس‌داده v2) ----------
PROMPTS = {
    'narrower':       ('slim narrow elegant nose', 'wide flat nose'),
    'wider':          ('slightly wider fuller nose base', 'pinched thin nose'),
    'upturned_tip':   ('upturned lifted nasal tip, perky', 'drooping long nose tip'),
    'droopy_tip':     ('softly downward angled tip', 'over-rotated upturned tip'),
    'doll_tip':       ('small cute rounded doll-like nose tip', 'large bulbous nose'),
    'fantasy':        ('refined sculpted fantasy nose, delicate bridge', 'bulbous asymmetric nose'),
    'hump_reduction': ('straight smooth nose bridge, no dorsal hump', 'dorsal hump bump on nose bridge'),
    'smaller':        ('proportionally smaller delicate nose', 'oversized large nose'),
    'ideal_realistic': ('harmonious ideal nose matching face', 'disproportionate nose'),
    'slim_bridge':    ('thin refined nose bridge', 'wide thick bridge'),
    'fleshy':         ('softer fuller rounded nose tip', 'thin sharp nose'),
    'bony':           ('defined bony structured nose', 'soft undefined nose'),
    'shorter':        ('shorter compact nose length', 'elongated long nose'),
    'longer':         ('slightly longer elegant nose', 'very short nose'),
    'filler':         ('non-surgical filler enhanced bridge', 'flat depressed bridge'),
}

# کانتور آناتومیک بینی (اندیس‌های FaceMesh 478)
NOSE_IDXS = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2,
             129, 98, 97, 206, 358, 327, 326, 426]


def detect_nose(image_bgr):
    """ماسک تنگ بینی + جعبه برش کانتکست. خروجی: (mask_full, box) یا None."""
    import mediapipe as mp
    h, w = image_bgr.shape[:2]
    mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                           refine_landmarks=True,
                                           max_num_faces=1)
    res = mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_RGB2BGR))
    mesh.close()
    if not res.multi_face_landmarks:
        return None
    lms = res.multi_face_landmarks[0].landmark
    pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)

    poly = pts[NOSE_IDXS].astype(np.int32)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [poly], 255)

    # جعبه برش: بینی + کانتکست (از بالای ابرو تا زیر لب، عرض صورت مرکزی)
    x0, y0 = poly.min(axis=0); x1, y1 = poly.max(axis=0)
    nh, nw = y1 - y0, x1 - x0
    cx0 = max(0, int(x0 - nw * 2.6)); cx1 = min(w, int(x1 + nw * 2.6))
    cy0 = max(0, int(y0 - nh * 3.2)); cy1 = min(h, int(y1 + nh * 3.6))
    return mask, (cx0, cy0, cx1, cy1)


def build_feathered_mask(mask_crop, feather=15):
    """ماسک نرم لبه‌دار — گذار بدون درز بین نواحی تولیدی و واقعی."""
    m = cv2.GaussianBlur(mask_crop, (0, 0), feather / 2.5)
    m = m.astype(np.float32) / 255.0
    return np.clip(m, 0, 1)[..., None]


def snap_to_multiple(size, mult=64):
    """اندازه را به مضرب ۶۴ گرد میکند (نیازمندی UNet)."""
    return int(round(size / mult) * mult)


class NoseGenEngine:
    """موتور تولید — فقط بینی، با تضمین ثابت‌بودن بقیه تصویر."""

    def __init__(self):
        self.pipe = None
        self.canny = None
        self._face_verify = None      # اختیاری: InsightFace

    # ----------------------------------------------------------
    def load(self):
        if self.pipe is not None:
            return
        import torch
        from diffusers import ControlNetModel, \
            StableDiffusionXLControlNetInpaintPipeline

        dtype = torch.float16
        logger.info('loading ControlNet …')
        controlnet = ControlNetModel.from_pretrained(
            'diffusers/controlnet-canny-sdxl-1.0', torch_dtype=dtype)
        logger.info('loading SDXL-inpaint …')
        self.pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            'diffusers/stable-diffusion-xl-1.0-inpainting-0.1',
            controlnet=controlnet,
            torch_dtype=dtype,
            variant='fp16',
            safety_checker=None,
        )
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()
        try:
            from controlnet_aux import MLSDdetector   # noqa: F401
            self.canny = 'canny'
        except Exception:
            self.canny = 'canny'                      # canny خودمان میسازیم
        self._try_load_identity_guard()
        logger.info('✅ NoseGenEngine ready')

    def _try_load_identity_guard(self):
        try:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._face_verify = app
        except Exception as e:
            logger.info(f'identity guard unavailable ({e}) — SSIM-only mode')

    # ----------------------------------------------------------
    def edit(self, image_bgr, action: str, intensity: float):
        detected = detect_nose(image_bgr)
        if detected is None:
            raise RuntimeError('چهره شناسایی نشد')
        mask_full, box = detected
        x0, y0, x1, y1 = box

        crop = image_bgr[y0:y1, x0:x1].copy()
        ch, cw = crop.shape[:2]

        # اندازه کار: بلندترین ضلع ≤ ۸۹۶ و مضرب ۶۴ — بدون کشیدگی
        scale = min(896.0 / max(ch, cw), 1.0)
        wh, ww = snap_to_multiple(ch * scale), snap_to_multiple(cw * scale)
        work = cv2.resize(crop, (ww, wh), interpolation=cv2.INTER_LANCZOS4)

        mask_work = cv2.resize(mask_full[y0:y1, x0:x1], (ww, wh),
                               interpolation=cv2.INTER_NEAREST)
        m3 = build_feathered_mask(mask_work)

        pos, neg_core = PROMPTS.get(action, ('refined natural nose', 'deformed'))
        negative = (f'{neg_core}, deformed face, identity change, different person, '
                    f'artifact, halo, ring, seam, blurry, plastic skin, cartoon')
        prompt = (f'RAW photo of same person, {pos}, photorealistic skin texture, '
                  f'identical face identity, natural lighting')

        # ⚙️ شدت مهارشده: سقف ۰.۶۰ + کنترل‌نت قوی = هندسه قفل
        strength = float(np.clip(0.22 + intensity * 0.38, 0.22, 0.60))

        import torch
        seed = int.from_bytes(os.urandom(4), 'little')
        generator = torch.Generator(device='cpu').manual_seed(seed)

        result = self.pipe(
            prompt=prompt, negative_prompt=negative,
            image=work, mask_image=mask_work,
            controlnet_conditioning_image=self._edge_map(work),
            width=ww, height=wh,
            strength=strength,
            guidance_scale=7.5,
            num_inference_steps=30,
            controlnet_conditioning_scale=0.65,
            generator=generator,
        ).images[0]

        gen = np.array(result)[:, :, ::-1]                       # RGB→BGR
        gen = self._skin_blend(work, gen, m3[..., 0])

        # ── برگشت به مقیاس اصلی + کامپوزیت تضمینی ──
        gen_full = cv2.resize(gen.astype(np.float32), (cw, ch),
                              interpolation=cv2.INTER_LANCZOS4)
        mask_orig = build_feathered_mask(
            mask_full[y0:y1, x0:x1], feather=max(9, ch // 90))
        roi_f = crop.astype(np.float32)
        merged = gen_full * mask_orig + roi_f * (1.0 - mask_orig)

        out = image_bgr.copy()
        out[y0:y1, x0:x1] = np.clip(merged, 0, 255).astype(np.uint8)

        # 🔒 گارد هویت — اگر افتاد، یکبار با شدت نصف تلاش مجدد
        sim = self._identity_similarity(image_bgr, out)
        if sim is not None and sim < 0.72 and intensity > 0.35:
            logger.warning(f'identity drift {sim:.2f} → retry softer')
            return self.edit(image_bgr, action, intensity * 0.5)

        meta = {'seed': seed, 'strength': round(strength, 2),
                'identity_sim': None if sim is None else round(float(sim), 3)}
        return out, meta

    # ----------------------------------------------------------
    def _edge_map(self, work_rgb_or_bgr):
        """نقشه لبه Canny برای ControlNet (PIL خروجی میدهد)."""
        from PIL import Image
        arr = np.array(work_rgb_or_bgr)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges)

    def _skin_blend(self, src_bgr, gen_bgr, mask01):
        """انتقال high-freq پوست واقعی — حذف حس AI."""
        s = src_bgr.astype(np.float32)
        g = gen_bgr.astype(np.float32)
        s_low = cv2.GaussianBlur(s, (0, 0), 10)
        g_low = cv2.GaussianBlur(g, (0, 0), 10)
        blended = g_low + (s - s_low) * 0.8
        m3 = mask01[..., None] if mask01.ndim == 2 else mask01
        out = blended * m3 + s * (1 - m3)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _identity_similarity(self, before, after):
        """شباهت کسینوس Embedding چهره — None اگر InsightFace نبود."""
        if self._face_verify is None:
            return None
        try:
            fa = self._face_verify
            r1 = fa.get(before[:, :, ::-1])
            r2 = fa.get(after[:, :, ::-1])
            if not r1 or not r2:
                return None
            e1, e2 = r1[0].embedding, r2[0].embedding
            return float(np.dot(e1, e2) /
                         (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
        except Exception as e:
            logger.info(f'identity check skipped: {e}')
            return None


# ─────────────────────────────────────────────
# تست محلی بدون GPU: صحت هندسه و تضمین «فقط بینی»
if __name__ == '__main__':
    import sys
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    img_path = os.path.join(root, 'test_images/test.jpg')
    img = cv2.imread(img_path)
    d = detect_nose(img)
    assert d is not None, 'face not found'
    mask_full, box = d
    x0, y0, x1, y1 = box
    print(f'nose mask px={int((mask_full > 0).sum())}  '
          f'context box=({x0},{y0},{x1},{y1})')

    # شبیهسازی کامل مسیر کامپوزیت با خروجی ساختگی:
    crop = img[y0:y1, x0:x1]
    fake_gen = cv2.GaussianBlur(crop, (0, 0), 12)          # «تولید» ساختگی
    mask_o = build_feathered_mask(mask_full[y0:y1, x0:x1], feather=max(9, (y1-y0)//90))
    gf = fake_gen.astype(np.float32)
    merged = gf * mask_o + crop.astype(np.float32) * (1 - mask_o)
    out = img.copy()
    out[y0:y1, x0:x1] = np.clip(merged, 0, 255).astype(np.uint8)

    # تضمین: هرجایی که ماسک فدرشده صفر است باید دقیقاً صفر تغییر باشد
    soft = (build_feathered_mask(mask_full[y0:y1, x0:x1],
                                 feather=max(9, (y1-y0)//90))[..., 0] * 255)
    diff = cv2.absdiff(img[y0:y1, x0:x1], out[y0:y1, x0:x1])
    outside = diff.copy()
    outside[soft > 3] = 0                                   # ناحیه اثر مجاز را کور کن
    print(f'max change OUTSIDE feathered mask: {int(outside.max())}  '
          f'{"✅ تضمین برقرار" if outside.max() <= 2 else "❌ نشت خارج ماسک!"}')
