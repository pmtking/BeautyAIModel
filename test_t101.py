# T-101 اعتبارسنجی نهایی — سنجه‌های درست + تست انتها-به-انتها فارسی
import sys, os, time
import numpy as np
import cv2

ROOT = '/home/mohammad/Desktop/project/BeautyAIModel'
sys.path.insert(0, os.path.join(ROOT, 'ai_training/src'))
sys.path.insert(0, os.path.join(ROOT, 'ai_training/src/models'))

from face_parser.model import FaceParserModel
from warping.nose_anatomy import NoseAnatomy
from warping.nose_styles import NoseAnatomyStyles
from warping.narrowing_pro import NarrowingPro

img = cv2.imread(os.path.join(ROOT, 'test_images/test.jpg'))
parser = FaceParserModel()
lm = parser.detect_from_image(img)
anat = NoseAnatomy(landmarks=lm, image_shape=img.shape)

PAD, TPL = 140, 56


def track(out, point):
    cx, cy = int(point[0]), int(point[1])
    tpl = img[cy-TPL//2:cy+TPL//2, cx-TPL//2:cx+TPL//2]
    if tpl.shape[0] < TPL or tpl.shape[1] < TPL:
        return None
    sy0, sy1 = max(0, cy-PAD), min(out.shape[0], cy+PAD)
    sx0, sx1 = max(0, cx-PAD), min(out.shape[1], cx+PAD)
    res = cv2.matchTemplate(out[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    return (loc[0] + TPL//2) - (cx - sx0), (loc[1] + TPL//2) - (cy - sy0)


print('═' * 62)
print(' بخش ۱ — مانورهای نوک (بال‌ها باید ثابت بمانند)')
print('═' * 62)
tip_cases = [
    ('upturned_tip i=0.7', lambda im: NoseAnatomyStyles.upturned_tip(im, lm, im.shape, 0.7), +8),
    ('upturned_tip i=1.0', lambda im: NoseAnatomyStyles.upturned_tip(im, lm, im.shape, 1.0), +8),
    ('droopy_tip   i=0.8', lambda im: NoseAnatomyStyles.droopy_tip(im, lm, im.shape, 0.8), -8),
]
panels = [cv2.resize(img, (700, int(700 * img.shape[0] / img.shape[1])))]
for name, fn, need in tip_cases:
    t0 = time.time(); out = fn(img.copy()); el = time.time() - t0
    dy = track(out, anat.get('tip'))[1]
    al = track(out, anat.get('alar_l')); ar = track(out, anat.get('alar_r'))
    up = -dy
    ok_lift = '✅' if (need > 0 and up >= need) or (need < 0 and up <= need) else '⚠️'
    ok_alar = '✅' if abs(al[0]) < 5 and abs(ar[0]) < 5 else '❌'
    print(f'{name}: lift={up:+3d}px {ok_lift}   alars dx=({al[0]:+d},{ar[0]:+d}) {ok_alar}   [{el:.2f}s]')
    panels.append(cv2.resize(out, panels[0].shape[:2][::-1]))

print()
print('═' * 62)
print(' بخش ۲ — باریک‌سازی (بال‌ها باید متقارن به داخل جمع شوند)')
print('═' * 62)
w0 = NarrowingPro.measure_shade_width(img, anat)
out_n = NoseAnatomyStyles.narrower(img.copy(), lm, img.shape, 0.7)
w1 = NarrowingPro.measure_shade_width(out_n, anat)
al = track(out_n, anat.get('alar_l')); ar = track(out_n, anat.get('alar_r'))
sym = abs(al[0] + ar[0])
inward = al[0] > 3 and ar[0] < -3
print(f'narrower i=0.7: shade width {w0:.0f} -> {w1:.0f}px  ({100*(w0-w1)/w0:+.1f}%)')
print(f'                alars dx=({al[0]:+d},{ar[0]:+d})  {"✅ داخل" if inward else "❌"}'
      f'  تقارن |dxL+dxR|={sym}px {"✅" if sym < 15 else "❌"}')
panels.append(cv2.resize(out_n, panels[0].shape[:2][::-1]))

print()
print('═' * 62)
print(' بخش ۳ — استایل ترکیبی (فقط تقارن مهم است)')
print('═' * 62)
for name, fn in [
    ('doll_tip  i=0.8', lambda im: NoseAnatomyStyles.doll_tip(im, lm, im.shape, 0.8)),
    ('fantasy   i=0.8', lambda im: NoseAnatomyStyles.fantasy(im, lm, im.shape, 0.8)),
]:
    t0 = time.time(); out = fn(img.copy()); el = time.time() - t0
    dy = track(out, anat.get('tip'))[1]
    al = track(out, anat.get('alar_l')); ar = track(out, anat.get('alar_r'))
    sym = abs(al[0] + ar[0])
    print(f'{name}: lift={-dy:+3d}px  alars dx=({al[0]:+d},{ar[0]:+d})'
          f'  تقارن={sym}px {"✅" if sym < 20 else "❌"}  [{el:.2f}s]')
    panels.append(cv2.resize(out, panels[0].shape[:2][::-1]))

while len(panels) < 6:
    panels.append(np.zeros_like(panels[0]))
grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:6])])
cv2.imwrite('/tmp/t101_grid.jpg', grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

print()
print('═' * 62)
print(' بخش ۴ — انتها-به-انتها: beauty_engine.process با متن فارسی')
print('═' * 62)
from beauty_engine.model import beauty_engine
texts = [
    'نوک بینی رو بالا ببر',
    'دماغم رو کوچیکتر کن',
    'بینی قلمی',
    'بینی عروسکی مزدانه',
    'قوز بینی رو بردار',
    'سوراخ بینی کوچیک',
]
crops = []
th, tw = 420, 420
cy_t, cx_t = int(anat.get('tip')[1]), int(anat.get('tip')[0])
def crop_nose(im):
    y0 = max(0, cy_t - tw // 2); x0 = max(0, cx_t - th // 2)
    return im[y0:y0 + th, x0:x0 + tw]
strip_orig = crop_nose(cv2.resize(img, (img.shape[1], img.shape[0])))
crops.append(strip_orig)
for t in texts:
    t0 = time.time()
    r = beauty_engine.process(img.copy(), t)
    el = time.time() - t0
    if r.get('status') != 'success':
        print(f'❌ "{t}" -> {r.get("message")}')
        continue
    desc = r.get('description', '')
    print(f'✅ "{t}" [{el:.2f}s] -> {desc}')
    crops.append(crop_nose(r['image']))
n = len(crops)
if n:
    # چیدمان ۳تایی در هر ردیف + پدینگ تا همه ردیفها همعرض شوند
    per = 3
    rows = []
    for i in range(0, n, per):
        row = crops[i:i + per]
        while len(row) < per:
            row.append(np.zeros_like(crops[0]))
        rows.append(np.hstack(row))
    width = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite('/tmp/t101_e2e.jpg', np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print('\nsaved: /tmp/t101_grid.jpg , /tmp/t101_e2e.jpg')
