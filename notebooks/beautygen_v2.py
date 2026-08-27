
!pip install -q diffusers==0.27.2 transformers accelerate peft safetensors
!pip install -q controlnet-aux insightface onnxruntime-gpu
!pip install -q mediapipe opencv-python-headless



import torch
from diffusers import (
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    AutoPipelineForImage2Image,
)
from controlnet_aux import MLSDdetector

DEVICE = "cuda"
DTYPE = torch.float16

# ── ControlNet برای حفظ هندسه صورت ──
controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0", torch_dtype=DTYPE)

pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    controlnet=controlnet,
    torch_dtype=DTYPE,
    variant="fp16",
    safety_checker=None,
).to(DEVICE)

# ── حافظه: T4 هم جواب میدهد ──
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()

# ── LoRA خودمان (بعد از آموزش) ──
# pipe.load_lora_weights("beuti-nose-lora-v2")



import cv2, numpy as np
from PIL import Image, ImageDraw

class ProNoseEditor:
    """ادیت بینی با Diffusion + قفل هویت + blend پوستی"""

    def __init__(self, pipe):
        self.pipe = pipe

    def nose_mask(self, image_pil, pad=0.06):
        """ماسک دقیق بینی از MediaPipe (روی تصویر اصلی)"""
        import mediapipe as mp
        arr = np.array(image_pil)
        h, w = arr.shape[:2]
        mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                refine_landmarks=True)
        res = mesh.process(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        if not res.multi_face_landmarks:
            # fallback: مستطیل مرکزی
            mask = Image.new("L", image_pil.size, 0)
            d = ImageDraw.Draw(mask)
            d.rectangle([int(w*.38), int(h*.32), int(w*.62), int(h*.58)], fill=255)
            return mask, None
        lms = res.multi_face_landmarks[0].landmark
        pts = np.array([[lm.x*w, lm.y*h] for lm in lms])
        # کانتور بینی: رادیکس تا زیر نوک + بال‌ها
        idxs = [168,6,197,195,5,4,1,19,94,2, 129,98,97,206, 358,327,326,426]
        poly = pts[idxs].astype(np.int32)
        mask_img = np.zeros((h,w), np.uint8)
        cv2.fillPoly(mask_img, [poly], 255)
        mask_img = cv2.dilate(mask_img, np.ones((31,31),np.uint8))
        mask = Image.fromarray(mask_img)
        return mask, pts

    def edit(self, image_bgr, action: str, intensity: float):
        pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        orig_size = pil.size
        work = pil.resize((1024, 1024))

        mask, _ = self.nose_mask(work)

        prompts = {
            "narrower":      ("slim narrow elegant nose", "wide flat nose"),
            "wider":         ("slightly wider fuller nose base", "pinched thin nose"),
            "upturned_tip":  ("upturned lifted nasal tip, perky",
                              "drooping long nose tip"),
            "droopy_tip":    ("softly downward angled tip", "over-rotated upturned tip"),
            "doll_tip":      ("small cute rounded doll-like nose tip",
                              "large bulbous nose"),
            "fantasy":       ("refined sculpted fantasy nose, delicate bridge",
                              "bulbous asymmetric nose"),
            "hump_reduction": ("straight smooth nose bridge, no dorsal hump",
                               "dorsal hump bump on nose bridge"),
            "smaller":       ("proportionally smaller delicate nose",
                              "oversized large nose"),
            "ideal_realistic":("harmonious ideal nose matching face",
                              "disproportionate nose"),
            "slim_bridge":   ("thin refined nose bridge", "wide thick bridge"),
            "fleshy":        ("softer fuller rounded nose tip", "thin sharp nose"),
            "bony":          ("defined bony structured nose", "soft undefined nose"),
            "shorter":       ("shorter compact nose length", "elongated long nose"),
            "longer":        ("slightly longer elegant nose", "very short nose"),
            "filler":        ("non-surgical filler enhanced bridge",
                             "flat depressed bridge"),
        }
        pos, neg_full = prompts.get(action, ("refined natural nose", "deformed"))
        negative = f"{neg_full}, deformed face, identity change, different person, " \
                   "artifact, halo, ring, seam, blurry, plastic skin, cartoon"

        strength = min(0.30 + intensity*0.45, 0.80)

        result = self.pipe(
            prompt=f"RAW photo of same person, {pos}, photorealistic skin texture, "
                   f"identical face identity, natural lighting",
            negative_prompt=negative,
            image=work,
            mask_image=mask,
            width=1024, height=1024,
            strength=strength,
            guidance_scale=8.0,
            num_inference_steps=34,
            controlnet_conditioning_scale=0.45,
        ).images[0]

        # ── blend پوستی: بافت واقعی پوست اطراف روی ناحیه تولیدی ──
        result = self._skin_blend(work, result, mask)

        return result.resize(orig_size)

    def _skin_blend(self, src, gen, mask):
        """انتقال high-freq پوست واقعی به خروجی — حذف حس AI"""
        s = np.array(src).astype(np.float32)
        g = np.array(gen).astype(np.float32)
        m = np.array(mask).astype(np.float32)/255.0

        # جداسازی low/high freq
        s_low = cv2.GaussianBlur(s, (0,0), 12)
        g_low = cv2.GaussianBlur(g, (0,0), 12)
        s_high = s - s_low           # بافت واقعی منبع

        blended = g_low + s_high*0.75   # فرم جدید + بافت قدیمی

        m3 = m[...,None]
        out = blended*m3 + s*(1-m3)
        return Image.fromarray(np.clip(out,0,255).astype(np.uint8))



def generate_best(editor, image_bgr, action, intensity, n=3):
    """۳ نمونه با seed مختلف → بهترین را انتخاب کن"""
    import random
    candidates = []
    for i in range(n):
        seed = random.randint(0, 10**9)
        torch.manual_seed(seed)
        out = editor.edit(image_bgr, action, intensity)
        score = quality_score(image_bgr, out)
        candidates.append((score, seed, out))

    candidates.sort(key=lambda x: -x[0])
    best_score, best_seed, best = candidates[0]
    print(f"selected seed={best_seed} score={best_score:.3f}")
    return best, candidates

def quality_score(before_bgr, after_bgr):
    """امتیاز ترکیبی: طبیعی بودن + حفظ هویت + بدون آرتیفکت"""
    b = cv2.cvtColor(before_bgr, cv2.COLOR_BGR2GRAY)
    a = cv2.cvtColor(after_bgr, cv2.COLOR_BGR2GRAY)
    if b.shape != a.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]))

    # ۱) حفظ هویت: شباهت SSIM در نواحی غیربینی (چشم/لب/پیشانی)
    h, w = b.shape
    keep = np.ones_like(b, bool)
    keep[int(h*.25):int(h*.65), int(w*.15):int(w*.85)] = False  # ناحیه بینی خارج
    ssim_keep = _ssim(b[keep], a[keep])

    # ۲) آرتیفکت: لبه‌های جدید ناخواسته
    eb = cv2.Canny(b, 50, 150); ea = cv2.Canny(a, 50, 150)
    new_edges = ((ea>0)&(eb==0))[int(h*.25):int(h*.70)].mean()

    # ۳) طبیعی بودن روشنایی: انحراف histogram محلی
    lb = cv2.Laplacian(b, cv2.CV_32F).var()
    la = cv2.Laplacian(a, cv2.CV_32F).var()
    texture_ratio = min(la/(lb+1e-6), lb/(la+1e-6))   # 1 = عالی

    score = ssim_keep - new_edges*0.5 - abs(1-texture_ratio)*0.3
    return float(score)

def _ssim(a, b):
    from skimage.metrics import structural_similarity as ssim_fn
    return ssim_fn(a.astype(np.uint8), b.astype(np.uint8))
