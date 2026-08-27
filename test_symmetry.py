# تست تقارن مطلق: ساخت چهره ۱۰۰٪ متقارن (نیمه+قرینه) و سنجش عدم تقارن narrowing
import sys, os
import numpy as np
import cv2

ROOT = '/home/mohammad/Desktop/project/BeautyAIModel'
sys.path.insert(0, os.path.join(ROOT, 'ai_training/src'))
sys.path.insert(0, os.path.join(ROOT, 'ai_training/src/models'))

from face_parser.model import FaceParserModel
from warping.nose_anatomy import NoseAnatomy
from warping.nose_styles import NoseAnatomyStyles

parser = FaceParserModel()


def track(img, out, point, tpl_s, pad):
    cx, cy = int(point[0]), int(point[1])
    t0x, t0y = cx - tpl_s // 2, cy - tpl_s // 2
    tpl = img[t0y:t0y+tpl_s, t0x:t0x+tpl_s]
    if tpl.shape[0] < tpl_s or tpl.shape[1] < tpl_s:
        return 0, 0
    sy0, sy1 = max(0, cy-pad), min(out.shape[0], cy+pad)
    sx0, sx1 = max(0, cx-pad), min(out.shape[1], cx+pad)
    res = cv2.matchTemplate(out[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    return (loc[0]+tpl_s//2)-(cx-sx0), (loc[1]+tpl_s//2)-(cy-sy0)


def run(path, label):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    # چهره متقارن مصنوعی: نیمه چپ + آینه آن
    half = img[:, :w//2]
    sym_img = np.hstack([half, half[:, ::-1]])
    lm = parser.detect_from_image(sym_img)
    if not lm or len(lm) < 468:
        print(f'{label}: شناسایی نشد')
        return
    a = NoseAnatomy(landmarks=lm, image_shape=sym_img.shape)
    out = NoseAnatomyStyles.narrower(sym_img.copy(), lm, sym_img.shape, 0.7)
    ts = max(28, min(56, int(a.nasal_width * 0.30)))
    pd = max(60, int(a.nasal_width * 1.6))
    dl = track(sym_img, out, a.get('alar_l'), ts, pd)
    dr = track(sym_img, out, a.get('alar_r'), ts, pd)
    asym = abs(dl[0] + dr[0])
    inward = dl[0] >= -3 and dr[0] <= 3
    print(f'{label}: alars dx=({dl[0]:+3d},{dr[0]:+3d})  '
          f'{"داخل✅" if inward else "بیرون❌"}  asym={asym}px '
          f'({100*asym/a.nasal_width:.1f}% عرض)  {"✅ متقارن" if asym <= max(8, 0.03*a.nasal_width) else "❌"}')
    return asym


run(os.path.join(ROOT, 'test_images/test.jpg'), 'متقارن‌شده(test)')
