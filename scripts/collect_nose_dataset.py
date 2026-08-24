#!/usr/bin/env python
"""
جمع‌آوری دیتاست استایل‌های بینی (قبل/بعد رینوپلاستی)
=====================================================
سه منبع:
  kaggle    — دانلود دیتاست‌های عمومی با kagglehub (نیاز به API key)
  manual    — ایمپورت عکس‌های آپلودی کاربر/کلینیک از یک پوشه
  synthetic — تولید جفت قبل/بعد با موتور آناتومیک خود پروژه

مثال:
  python scripts/collect_nose_dataset.py --source synthetic --count 200
  python scripts/collect_nose_dataset.py --source manual --dir ~/Pictures/cases
  python scripts/collect_nose_dataset.py --source kaggle --query rhinoplasty before after
"""
import argparse
import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJ / 'datasets' / 'nose_styles'
RAW = DATA_ROOT / 'raw'
CSV_PATH = DATA_ROOT / 'pairs.csv'

STYLES = ['doll_tip', 'fantasy', 'half_fantasy', 'fleshy', 'bony',
          'upturned_tip', 'filler', 'slim_bridge', 'hump_reduction',
          'narrower', 'shorter', 'natural']

VIEWS = ['front', 'left_profile', 'right_profile']


def ensure_dirs():
    RAW.mkdir(parents=True, exist_ok=True)


def append_pair(pair_id: str, style: str, view: str, source: str):
    new = not CSV_PATH.exists()
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['pair_id', 'style', 'view', 'source', 'created'])
        w.writerow([pair_id, style, view, source, date.today().isoformat()])


# =========================================================
#   SYNTHETIC — تولید جفت قبل/بعد با پارامتر معلوم
# =========================================================

def gen_synthetic(count: int, seed: int = 42):
    """
    از تصاویر موجود در test_images یا datasets، نسخه «بعد» را با
    موتور آناتومیک می‌سازیم. چون شدت اعمال‌شده را خودمان انتخاب کرده‌ایم،
    این جفت‌ها ground-truth کالیبراسیون هستند.
    """
    import cv2
    import numpy as np

    src_dir = PROJ / 'test_images'
    sources = sorted(list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png')))
    if not sources:
        print('❌ هیچ تصویر منبعی در test_images نیست')
        return

    sys.path.insert(0, str(PROJ / 'ai_training' / 'src' / 'models'))
    from beauty_engine.model import beauty_engine  # noqa: E402

    rng = random.Random(seed)
    made = 0
    for i in range(count):
        style = STYLES[i % len(STYLES)]
        intensity = round(rng.uniform(0.45, 0.9), 2)
        view = VIEWS[0] if len(VIEWS) == 1 else rng.choice(VIEWS)

        src_path = rng.choice(sources)
        img = cv2.imread(str(src_path))
        if img is None:
            continue
        # کوچک برای سرعت
        h, w = img.shape[:2]
        if max(h, w) > 1200:
            s = 1200.0 / max(h, w)
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

        landmarks = beauty_engine.face_parser.detect_from_image(img)
        if not landmarks:
            continue

        from warping.nose_styles import _resolve, NoseAnatomyStyles
        anat = _resolve(landmarks, img.shape)

        pair_id = f'syn_{i:04d}'
        out_dir = RAW / pair_id
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / 'before.jpg'), img)

        try:
            edited = NoseAnatomyStyles.__dict__.get(style)(
                img.copy(), landmarks, img.shape, intensity)
        except Exception as e:
            print(f'⚠️ {style} failed: {e}')
            continue

        cv2.imwrite(str(out_dir / 'after.jpg'), edited)
        meta = {
            'pair_id': pair_id,
            'style': style,
            'intensity_applied': intensity,
            'view': view,
            'source': 'synthetic:anatomy-engine',
            'consent': True,
            'created': date.today().isoformat(),
        }
        (out_dir / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        append_pair(pair_id, style, view, 'synthetic')
        made += 1
        if made % 20 == 0:
            print(f'… {made}/{count}')

    print(f'✅ {made} جفت سینتکی ساخته شد → {RAW}')


# =========================================================
#   MANUAL — ایمپورت از پوشه کاربر
# =========================================================

def import_manual(src_dir: Path, default_style: str = 'unlabeled'):
    import shutil
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    files = [p for p in src_dir.rglob('*') if p.suffix.lower() in exts]
    # جفت‌های before/after که کنار هم هستند
    pairs = {}
    for p in files:
        name = p.stem.lower()
        pid = None
        for token in ('before', 'after'):
            if token in name:
                pid = name.replace(token, '').strip('_- ')
                break
        if pid is None:
            continue
        pairs.setdefault(pid, {})[
            'before.jpg' if 'before' in p.stem.lower() else 'after.jpg'] = p

    made = 0
    for pid, group in sorted(pairs.items()):
        if 'before.jpg' not in group or 'after.jpg' not in group:
            continue
        out_dir = RAW / f'user_{pid}'[:60]
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(group['before.jpg'], out_dir / 'before.jpg')
        shutil.copy(group['after.jpg'], out_dir / 'after.jpg')
        meta = {
            'pair_id': pid, 'style': default_style,
            'view': 'front', 'source': 'manual-upload',
            'consent': False,  # ← باید فرم رضایت جمع شود
            'created': date.today().isoformat(),
        }
        (out_dir / 'meta.json').write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        append_pair(pid, default_style, 'front', 'manual')
        made += 1
    print(f'✅ {made} جفت دستی ایمپورت شد')


# =========================================================
#   KAGGLE — دانلود عمومی (نیاز به kagglehub + API key)
# =========================================================

def fetch_kaggle(query: str):
    try:
        import kagglehub
    except ImportError:
        print('❌ pip install kagglehub و قرار دادن kaggle.json لازم است')
        return
    print(f'⏳ جستجو/دانلود: {query}')
    path = kagglehub.dataset_download(query)
    print(f'✅ دانلود شد → {path}\nحالا با --source manual --dir "{path}" واردش کن')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', choices=['synthetic', 'manual', 'kaggle'],
                    required=True)
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--dir', type=str, default='')
    ap.add_argument('--query', type=str, default='rhinoplasty before after')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    ensure_dirs()
    if args.source == 'synthetic':
        gen_synthetic(args.count, args.seed)
    elif args.source == 'manual':
        if not args.dir:
            print('❌ --dir لازم است')
            return
        import_manual(Path(args.dir).expanduser())
    else:
        fetch_kaggle(args.query)


if __name__ == '__main__':
    main()
