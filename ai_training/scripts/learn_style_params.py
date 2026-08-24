#!/usr/bin/env python
"""
یادگیری پارامترهای استایل‌های بینی از دیتاست قبل/بعد
====================================================
ورودی: datasets/nose_styles/raw/<pair>/{before.jpg, after.jpg, meta.json}
خروجی: datasets/nose_styles/learned_params.json

برای هر جفت:
  1. FaceMesh روی before و after
  2. استخراج بردار تغییر ۲۰ نقطه آناتومیک:
       tip_lift_ratio      جابجایی نوک / ارتفاع بینی
       alar_width_ratio    عرض بعد / عرض قبل
       dorsum_width_ratio  عرض قوس بعد/قبل
       radix_fill          جابجایی رادیکس
       columella_shift     جابجایی کولوملا / ارتفاع
  3. میانگین وزنی هر استایل → فایل پارامتر

سپس موتور warp (nose_styles.py) می‌تواند این ضرایب واقعی را بخواند.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJ = Path(__file__).resolve().parent.parent.parent   # ← ریشه پروژه (BeautyAIModel)
sys.path.insert(0, str(PROJ / 'ai_training' / 'src' / 'models'))
sys.path.insert(0, str(PROJ / 'backend'))


def measure_pair(before_lms, after_lms, shape) -> dict:
    """بردار تغییر آناتومیک بین دو ست لندمارک (نرمال‌شده)."""
    from warping.nose_styles import _resolve  # type: ignore
    h, w = shape[:2]

    a = _resolve(before_lms, shape)
    b = _resolve(after_lms, shape)
    if not a or not b or not a.valid or not b.valid:
        return {}

    height_a = max(a.nasal_height, 1e-6)
    width_a = max(a.nasal_width, 1e-6)

    tip_a, tip_b = a.get('tip'), b.get('tip')
    radix_a, radix_b = a.get('radix'), b.get('radix')
    col_a, col_b = a.get('columella'), b.get('columella')

    return {
        'tip_lift': float((tip_a[1] - tip_b[1]) / height_a),
        'radix_shift': float(np.linalg.norm(radix_b - radix_a) / height_a),
        'columella_shift': float((col_a[1] - col_b[1]) / height_a),
        'alar_width_ratio': float(b.nasal_width / width_a),
        'height_ratio': float(b.nasal_height / height_a),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=str(PROJ / 'datasets' / 'nose_styles'))
    ap.add_argument('--out', default='')
    args = ap.parse_args()

    data = Path(args.data)
    raw = data / 'raw'
    out_path = Path(args.out) if args.out else data / 'learned_params.json'

    sys.path.insert(0, str(PROJ / 'backend'))
    from app.services.face_mesh_robust import robust_face_mesh

    buckets = defaultdict(list)
    pairs = sorted(raw.glob('*/meta.json'))
    print(f'📁 {len(pairs)} جفت پیدا شد')

    for meta_path in pairs:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        style = meta.get('style')
        folder = meta_path.parent
        before_p, after_p = folder / 'before.jpg', folder / 'after.jpg'
        if not style or not before_p.exists() or not after_p.exists():
            continue

        before = cv2.imread(str(before_p))
        after = cv2.imread(str(after_p))
        if before is None or after is None:
            continue

        lb, _ = robust_face_mesh.detect(before)
        la, _ = robust_face_mesh.detect(after)
        if not lb or not la:
            continue

        m = measure_pair(lb, la, before.shape)
        if m:
            m['intensity'] = meta.get('intensity_applied', 0.6)
            buckets[style].append(m)

    # میانگین‌گیری
    learned = {}
    for style, items in buckets.items():
        agg = {}
        for key in items[0]:
            vals = [it[key] for it in items if key in it]
            agg[key] = {
                'mean': round(float(np.mean(vals)), 4),
                'std': round(float(np.std(vals)), 4),
                'n': len(vals),
            }
        learned[style] = agg

    out_path.write_text(json.dumps(learned, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    print(f'✅ پارامترهای یادگرفته‌شده → {out_path}')
    for style, agg in learned.items():
        print(f"  {style:15} n={agg['tip_lift']['n']:3}  "
              f"tip_lift={agg['tip_lift']['mean']:+.3f}  "
              f"alar×{agg['alar_width_ratio']['mean']:.3f}")


if __name__ == '__main__':
    main()
