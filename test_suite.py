#!/usr/bin/env python3
"""
🧪 راستی‌آزمایی کامل موتور پردازش چهره BeautyAIModel
تست‌ها:
  A) درک متن فارسی → تشخیص صحیح area/action
  B) خروجی تصویر سالم (بدون ناحیه سیاه/پارگی)
  C) تغییر موضعی (فقط ناحیه هدف عوض شود — بدون آرتیفکت سراسری)
  D) پایداری (دو بار اجرا → نتیجه یکسان)
  E) شدت‌های مختلف (0.3 / 0.7 / 1.0)
"""
import json, sys, time, base64, urllib.request, urllib.error, uuid
import numpy as np
import cv2

BASE = 'http://localhost:8000'
IMG = 'test_images/test.jpg'
raw = open(IMG, 'rb').read()
print(f'🖼️  input: {IMG} ({len(raw)//1024} KB)\n')

def post_3d(text, intensity=0.7):
    boundary = uuid.uuid4().hex
    parts = []
    def field(name, val):
        nonlocal parts
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{val}\r\n'.encode())
    field('text', text); field('intensity', str(intensity))
    binpart = f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="t.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode() + raw + b'\r\n'
    body = b''.join(parts[:-1]) if False else None
    # ترتیب: text, intensity, file
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'.encode()
            + f'--{boundary}\r\nContent-Disposition: form-data; name="intensity"\r\n\r\n{intensity}\r\n'.encode()
            + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="t.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()
            + raw + f'\r\n--{boundary}--\r\n'.encode())
    req = urllib.request.Request(BASE + '/api/v1/3d-filter', data=body,
                                 headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    d['_latency'] = time.time() - t0
    return d

orig = cv2.imread(IMG)
oh, ow = orig.shape[:2]
ORIG_SMALL = cv2.resize(orig, (900, int(900*oh/ow)))

def analyze(out_b64, label):
    """بررسی سلامت خروجی + میزان/موقعیت تغییر"""
    img = cv2.imdecode(np.frombuffer(base64.b64decode(out_b64), np.uint8), cv2.IMREAD_COLOR)
    issues = []
    if img is None:
        return {'ok': False, 'issues': ['تصویر decode نشد']}
    if img.shape[:2] != (oh, ow):
        issues.append(f'ابعاد عوض شده {img.shape[:2]} != {(oh,ow)}')
    small = cv2.resize(img, (900, ORIG_SMALL.shape[0]))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    black = (gray < 8).mean()*100
    white = (gray > 250).mean()*100
    if black > 1.0: issues.append(f'{black:.1f}% پیکسل سیاه (پارگی warp)')
    if white > 8.0: issues.append(f'{white:.1f}% پیکسل سفید محض')
    d = cv2.absdiff(small, ORIG_SMALL).mean(axis=2)
    changed = (d > 6).astype(np.uint8)
    pct = changed.mean()*100
    ys, xs = np.where(changed)
    box = (xs.min(), xs.max(), ys.min(), ys.max()) if len(xs) else None
    spread = (pct > 40)  # تغییر سراسری = احتمال اعوجاج کل صورت
    return {'ok': len(issues) == 0, 'issues': issues, 'changed_pct': round(pct,2),
            'bbox': box, 'global_distortion': spread}

PASS, FAIL = 0, 0
def report(name, cond, detail=''):
    global PASS, FAIL
    mark = '✅' if cond else '❌'
    cond and (PASS := PASS+1) or (not cond and (FAIL := FAIL+1))
    print(f'   {mark} {name}' + (f' — {detail}' if detail else ''))

# ---------- A) درک زبان فارسی ----------
print('━'*62)
print('A) درک متن فارسی (NLU)')
cases = [
    ('نوک بینی بالا باشه',            ('nose', 'upturned_tip')),
    ('بینی کوچیک تر بشه',             ('nose', 'smaller')),
    ('قوز بینی رو بردار',             ('nose', 'hump_reduction')),
    ('بینی عروسکی دوست دارم',         ('nose', 'doll_tip')),
    ('بینی قلمی و کشیده باشه',        ('nose', 'slim_bridge')),
    ('بینی پهن تر بشه',               ('nose', 'wider')),
    ('لب روسی بزن برام',              ('lip',  'russian')),
    ('چونه کوچک تر بشه',              ('chin', None)),
    ('گونه ها پرتر بشن',              ('cheek', None)),
    ('فک رو زاویه دار کن',            ('jaw',  None)),
]
nlu_results = []
for text, (exp_area, exp_action) in cases:
    try:
        r = post_3d(text)
        ch = r.get('changes', {})
        area_ok = ch.get('area') == exp_area
        acts = [ch.get('action')] if not isinstance(ch.get('actions'), list) else ch['actions']
        act_ok = (exp_action is None) or (exp_action in acts or ch.get('style') == exp_action)
        nlu_results.append((text, r, area_ok and act_ok))
        det = f"area={ch.get('area')} action={acts or ch.get('style')}"
        report(f'«{text}»', area_ok and act_ok, det + f' [{r["_latency"]:.1f}s]')
    except Exception as e:
        report(f'«{text}»', False, f'EXC: {e}')
        nlu_results.append((text, None, False))

# ---------- B/C) کیفیت بصری روی چند استایل ----------
print('━'*62)
print('B/C) کیفیت بصری خروجی (آرتیفکت / موضعی بودن تغییر)')
vis_cases = ['نوک بینی بالا', 'بینی کوچک تر', 'عروسکی', 'قوز بینی', 'فانتزی']
for i, text in enumerate(vis_cases):
    inten = [0.4, 0.7, 1.0][i % 3]
    try:
        r = post_3d(text, inten)
        b64 = r.get('filtered_image') or r.get('image')
        res = analyze(b64, text)
        ok = res['ok'] and not res['global_distortion']
        det = f"changed={res['changed_pct']}% bbox={res['bbox']}"
        if res['issues']: det += ' ⚠️ ' + '; '.join(res['issues'])
        report(f'«{text}» @{inten}', ok, det)
    except Exception as e:
        report(f'«{text}» @{inten}', False, f'EXC: {e}')

# ---------- D) پایداری ----------
print('━'*62)
print('D) پایداری (determinism)')
try:
    r1 = post_3d('نوک بینی بالا', 0.6)
    r2 = post_3d('نوک بینی بالا', 0.6)
    a1 = cv2.imdecode(np.frombuffer(base64.b64decode(r1['filtered_image']), np.uint8), 0)
    a2 = cv2.imdecode(np.frombuffer(base64.b64decode(r2['filtered_image']), np.uint8), 0)
    same = a1.shape == a2.shape and np.abs(a1.astype(int)-a2.astype(int)).mean() < 0.5
    report('دو اجرای یکسان → خروجی یکسان', same,
           f'diff={np.abs(a1.astype(int)-a2.astype(int)).mean():.3f}' if same else 'خروجی متفاوت!')
except Exception as e:
    report('دو اجرای یکسان → خروجی یکسان', False, f'EXC: {e}')

# ---------- E) endpoint های دیگر ----------
print('━'*62)
print('E) سایر endpoint ها')
for path in ['/health', '/api/v1/edit/styles', '/api/v1/3d-test']:
    try:
        with urllib.request.urlopen(BASE + path, timeout=10) as r:
            j = json.loads(r.read())
            report(path, True, str(j)[:60])
    except Exception as e:
        report(path, False, str(e)[:60])

print('━'*62)
print(f'📊 نتیجه: {PASS} ✅ | {FAIL} ❌')
sys.exit(1 if FAIL else 0)
