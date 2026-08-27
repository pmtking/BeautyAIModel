"""
⚡ Generative Engine v2 — SDXL + ControlNet + Skin-blend + Ensemble
فعال فقط وقتی GPU باشد؛ روتر هیبرید خودش تصمیم میگیرد.
"""
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class GenerativeEngine:
    _instance = None
    _available = None

    @classmethod
    def get(cls):
        if cls._instance is None and cls._check_gpu():
            try:
                cls._instance = cls._load()
            except Exception as e:
                logger.warning(f"generative load failed: {e}")
                cls._instance = None
        return cls._instance

    @classmethod
    def _check_gpu(cls):
        if cls._available is None:
            try:
                import torch
                cls._available = torch.cuda.is_available()
            except ImportError:
                cls._available = False
        return cls._available

    @classmethod
    def _load(cls):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                        '..', 'notebooks'))
        import torch
        free, total = torch.cuda.mem_get_info()
        # حداقل VRAM قابل تنظیم — پیشفرض 10GB (RTX 3060/570 12GB پاس میشود)
        # برای GPU کوچکتر: متغیر محیطی MIN_GEN_VRAM_GB را کمتر کن
        min_gb = float(os.environ.get('MIN_GEN_VRAM_GB', 10))
        if total < min_gb * 10**9:
            logger.info(f"GPU {total/10**9:.1f}GB < {min_gb}GB — generative disabled")
            return None

        exec(open(os.path.join(os.path.dirname(__file__), '..', '..',
                               '..', 'notebooks', 'beautygen_v2.py')).read(),
             {'__name__': '__gen_v2__'})
        editor = ProNoseEditor(pipe)      # noqa — از beautygen_v2
        cls.editor = editor
        return editor

    @classmethod
    def edit(cls, image_bgr, action, intensity):
        eng = cls.get()
        if eng is None:
            raise RuntimeError("generative unavailable")
        best, all_c = generate_best(cls.editor, image_bgr, action, intensity, n=3)  # noqa
        return best
