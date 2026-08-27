
!pip install -q diffusers==0.27.2 transformers accelerate peft controlnet-aux safetensors
!pip install -q bitsandbytes datasets

#CELL2

import json, random
from pathlib import Path

DS = Path("datasets/nose_styles")
records = []
for line in open(DS/"manifest.jsonl"):
    r = json.loads(line)
    if r["split"] != "train": continue
    meta = json.loads(open(r["meta"]).read())
    records.append({
        "before": r["before"],
        "after": r["after"],
        "style": meta.get("style", ""),
        "intensity": meta.get("intensity_applied", 0.5),
        # پرامپت فارسی+انگلیسی برای هر استایل
        "prompt_en": {
            "narrower": "slimmer narrower nose, natural look",
            "wider": "slightly wider nose base",
            "upturned_tip": "upturned nasal tip, lifted",
            "droopy_tip": "downward drooping nasal tip",
            "doll_tip": "cute doll-like small nose, rounded tip",
            "fantasy": "elegant fantasy nose shape, refined",
            "hump_reduction": "smooth nose bridge without hump",
            "smaller": "smaller overall nose size",
            "natural": "very subtle natural refinement",
            "ideal_realistic": "ideal harmonious nose for this face",
            "filler": "non-surgical filler augmented nose bridge",
            "slim_bridge": "thin slim nose bridge",
            "fleshy": "fleshy fuller nose tip",
            "bony": "bony defined nose structure",
            "shorter": "shorter nose length",
            "longer": "longer nose length",
        }.get(meta.get("style",""), "refined nose"),
    })

random.seed(42); random.shuffle(records)
print(f"train pairs: {len(records)}")

# caption file برای diffusers
import shutil
out = Path("training_data"); out.mkdir(exist_ok=True)
for i, rec in enumerate(records):
    (out/f"{i:05d}.txt").write_text(
        f"photo of face, {rec['prompt_en']}, high quality, realistic"
    )
shutil.copy(rec["after"], out/f"{i:05d}.png") if False else None

#CELL3

import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json

MODEL = "runwayml/stable-diffusion-inpainting"
device = "cuda"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    MODEL, torch_dtype=torch.float16, safety_checker=None)
pipe.to(device)

unet = pipe.unet
lora_cfg = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    lora_dropout=0.05, bias="none",
)
unet = get_peft_model(unet, lora_cfg)
unet.print_trainable_parameters()
# ~40M trainable — سریع و کم‌حجم

class PairDataset(Dataset):
    def __init__(self, records, size=512):
        self.recs = records; self.size = size
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        before = Image.open(r["before"]).convert("RGB").resize((self.size,)*2)
        after  = Image.open(r["after"]).convert("RGB").resize((self.size,)*2)
        # mask = ناحیه بینی از لندمارک (ساده: مستطیل مرکزی)
        mask = Image.new("L", (self.size,)*2, 0)
        from PIL import ImageDraw
        d = ImageDraw.Draw(mask)
        d.rectangle([self.size*0.35, self.size*0.30,
                     self.size*0.65, self.size*0.62], fill=255)
        return before, after, mask, r["prompt_en"]

# حلقه آموزش ساده (در عمل: accelerate + gradient checkpointing)
# full script در train_lora.py
