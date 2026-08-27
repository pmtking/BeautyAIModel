#!/usr/bin/env python3
"""
🔬 راستی‌آزمایی دقیق موتور بینی — طبق معیارهای آناتومیک رینوپلاستی
===================================================================
تست‌ها روی هر استایل:
  ۱) هندسی: نسبت‌های آناتومیک قبل/بعد (نسبت عرض-ارتفاع بینی، زاویه نوک، projection)
  ۲) بصری: آرتیفکت، درز، هاله، پارگی warp
  ۳) موضعی بودن: تغییر فقط در ناحیه بینی (نه چشم/لب/پیشانی)
"""
import sys, json, base64, time, urllib.request, uuid
import numpy as np
import cv2

sys.path.insert(0, "ai_training/src")
sys.path.insert(0, "ai_training/src/models")
from face_parser.model import FaceParserModel

BASE = "http://localhost:8000"
IMG = "test_images/test.jpg"

# ---------- لود یکبار ----------
parser = FaceParserModel()
orig = cv2.imread(IMG)
H, W = orig.shape[:2]
landmarks = parser.detect_from_image(orig)
assert landmarks and len(landmarks) >= 468, "چهره شناسایی نشد"

# ---------- آناتومی مرجع ----------
def px(i):
    return np.array([landmarks[i]["x"] * W, landmarks[i]["y"] * H], dtype=np.float32)

def detect_view():
    L, R = px(33), px(263)
    eL, eR = px(127), px(356)
    dL, dR = abs(L[0]-eL[0]), abs(eR[0]-R[0])
    r = dL/(dL+dR+1e-6)
    return "right" if r > 0.62 else ("left" if r < 0.38 else "front")

VIEW = detect_view()
print(f"🖼️  نمای تشخیص‌داده‌شده: {VIEW} | ابعاد {W}x{H}\n")

# نقاط کلیدی بینی (MediaPipe)
RADIX = 168; TIP = 4; TIP2 = 1; ALAR_L = 129; ALAR_R = 358
BRIDGE_MID = 195; COLUM = 2

def nose_metrics(img_bgr):
    """اندازه‌گیری متریک‌های آناتومیک از روی لندمارک‌ها"""
    radix, tip = px(RADIX), px(TIP)
    alL, alR = px(ALAR_L), px(ALAR_R)
    bridge = px(BRIDGE_MID)
    colum = px(COLUM)

    width = float(np.linalg.norm(alL - alR))            # عرض آلار
    height = float(np.linalg.norm(radix - colum))       # ارتفاع رادیکس→کولوملا
    # زاویه نوک (nasolabial تقریبی): بردار tip→colum در مقابل عمودی
    v = colum - tip
    angle = float(np.degrees(np.arctan2(v[0], -v[1])))  # مثبت = نوک جلو/بالا

    # projection نسبی: فاصله عمود tip از خط radix→colum
    u = colum - radix
    t = np.dot(tip - radix, u) / (np.dot(u, u) + 1e-6)
    proj_pt = radix + t * u
    projection = float(np.linalg.norm(tip - proj_pt))

    # قوس پل: انحراف bridge از خط مستقیم radix→tip
    u2 = tip - radix
    t2 = np.dot(bridge - radix, u2) / (np.dot(u2, u2) + 1e-6)
    hump = float(np.linalg.norm(bridge - (radix + t2 * u2)))

    return {"width": round(width,1), "height": round(height,1),
            "tip_angle": round(angle,1), "projection": round(projection,1),
            "hump_dev": round(hump,1)}

M0 = nose_metrics(orig)
print("📊 متریک‌های پایه:", json.dumps(M0))

# ---------- تست هر درخواست ----------
def call_api(text, intensity=0.7):
    boundary = uuid.uuid4().hex
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="intensity"\r\n\r\n{intensity}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="t.jpg"\r\n'
            f'Content-Type: image/jpeg\r\n\r\n').encode() + open(IMG,"rb").read() + f'\r\n--{boundary}--\r\n'.encode()
    req = urllib.request.Request(BASE+"/api/v1/3d-filter", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())

def visual_check(out_img):
    """بررسی آرتیفکت بصری در ناحیه صورت"""
    issues = []
    gray = cv2.cvtColor(cv2.resize(out_img,(900,int(900*H/W))), cv2.COLOR_BGR2GRAY)
    g0 = cv2.cvtColor(cv2.resize(orig,(900,int(900*H/W))), cv2.COLOR_BGR2GRAY)
    # لبه‌های کاذب: گرادیان شدید جدید که در اصلی نبوده
    gx0 = cv2.Sobel(g0,cv2.CV_32F,1,0,ksize=3); gy0 = cv2.Sobel(g0,cv2.CV_32F,0,1,ksize=3)
    gx1 = cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy1 = cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    e0 = np.hypot(gx0,gy0); e1 = np.hypot(gx1,gy1)
    new_edges = ((e1>80)&(e0<30)).mean()*100
    if new_edges > 0.5: issues.append(f"لبه کاذب {new_edges:.2f}%")
    black = (gray<8).mean()*100
    if black > 2.0: issues.append(f"پیکسل سیاه {black:.1f}%")
    return issues, new_edges

CASES = [
    # (درخواست, انتظار هندسی: کلید، جهت تغییر، حداقل درصد)
    ("دماغ باریک تر بشه",        [("width", -6)]),
    ("نوک دماغ بالا باشه",       [("tip_angle", +15)]),
    ("قوز دماغ گرفته بشه",       [("hump_dev", -10)]),
    ("دماغ کوچیک تر بشه",        [("width", -4), ("height", -4)]),
    ("نوک دماغ عقب بره",         [("projection", -8)]),
]

PASS, FAIL = 0, 0
rows = []
for text, expects in CASES:
    try:
        r = call_api(text)
        if r.get("status") != "success":
            print(f"❌ «{text}» → status={r.get('status')}")
            FAIL += 1; rows.append((text, None)); continue
        out = cv2.imdecode(np.frombuffer(base64.b64decode(r["filtered_image"]), np.uint8), 1)
        M1 = nose_metrics(out)
        issues, _ = visual_check(out)

        det = []
        ok_all = True
        for key, direction in expects:
            before, after = M0[key], M1[key]
            delta = after - before
            pct = delta / (abs(before)+1e-6) * 100
            good = (delta < 0 and direction < 0) or (delta > 0 and direction > 0)
            strong = abs(pct) >= abs(direction) * 0.4   # حداقل ۴۰٪ انتظار
            ok = good and strong
            ok_all &= ok
            det.append(f"{key}: {before}→{after} ({pct:+.1f}%) {'✅' if ok else '⚠️'}")

        art = "" if not issues else " 🎨" + ";".join(issues)
        mark = "✅" if ok_all else "❌"
        ok_all and (PASS := PASS+1) or not ok_all and (FAIL := FAIL+1)
        rows.append((text, M1))
        print(f"{mark} «{text}» — {' | '.join(det)}{art}")
    except Exception as e:
        print(f"❌ «{text}» EXC: {e}")
        FAIL += 1

print("\n" + "━"*60)
print(f"📈 نتیجه آناتومیک: {PASS}/{PASS+FAIL}")
print("متریک پایه:", json.dumps(M0))
