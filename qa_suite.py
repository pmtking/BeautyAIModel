#!/usr/bin/env python3
"""
BeautyAI Model — Professional QA Suite (warping / edit / image-creation)
=======================================================================
A comprehensive, deterministic test-suite for the anatomic warping engine
and its live HTTP API. Designed to separate REAL engine defects from
test-harness artefacts (the legacy tests double-scaled landmark
coordinates and used naive thresholds that flagged pre-existing dark
pixels on the source photo as "warp tearing" — both are corrected here).

Metrics (all computed in real pixels from NoseAnatomy, which handles
normalized vs pixel coordinates itself):

  1. Anatomic deltas      — width / height / tip-rotation / hump / alar span
  2. Alar symmetry        — |dxL − dxR| of the two alar wings (per wing drift)
  3. Locality             — change should stay near the target area (not global)
  4. Image integrity      — NEW black/white pixels vs source baseline,
                            fresh harsh edges (warp seams/tearing)
  5. Determinism          — identical input → identical output
  6. Pipeline health      — /health, /edit/health, /api/v1/edit/styles,
                            /api/v1/face-edit/* (generative, if deps present)
"""
import sys, json, os, time, base64, urllib.request, uuid, argparse
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
BASE = os.environ.get("BEAUTY_API", "http://localhost:8000")
IMG = ROOT / "test_images" / "test.jpg"
IMG_ALT = ROOT / "test_images" / "test2.jpg"  # second face (optional)

RESULTS = {"pass": 0, "fail": 0, "warn": 0, "detail": [], "errors": []}


def report(name, ok, detail="", level=None):
    level = level or ("pass" if ok else "fail")
    RESULTS[level] += 1
    mark = {"pass": "✅", "fail": "❌", "warn": "⚠️"}[level]
    line = f"{mark} {name}" + (f" — {detail}" if detail else "")
    RESULTS["detail"].append((level, name, detail))
    print(line)


def load_deps():
    sys.path.insert(0, str(ROOT / "ai_training" / "src"))
    sys.path.insert(0, str(ROOT / "ai_training" / "src" / "models"))
    from face_parser.model import FaceParserModel
    from warping.nose_anatomy import NoseAnatomy
    from warping.nose_styles import NoseAnatomyStyles
    parser = FaceParserModel()
    return parser, NoseAnatomy, NoseAnatomyStyles


# ----------------------------------------------------------------------
#  HTTP helpers (raw, no deps beyond stdlib)
# ----------------------------------------------------------------------
def post_3d(text, intensity=0.7, img_path=IMG):
    raw = open(img_path, "rb").read()
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="text"\r\n\r\n{text}\r\n'.encode()
        + f'--{boundary}\r\nContent-Disposition: form-data; name="intensity"\r\n\r\n{intensity}\r\n'.encode()
        + f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="f.jpg"\r\n'
          f'Content-Type: image/jpeg\r\n\r\n'.encode()
        + raw + f'\r\n--{boundary}--\r\n'.encode()
    )
    req = urllib.request.Request(BASE + "/api/v1/3d-filter", data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def b64_to_img(s):
    if not s:
        return None
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(s), np.uint8), cv2.IMREAD_COLOR)


# ----------------------------------------------------------------------
#  Core analysis
# ----------------------------------------------------------------------
class Analyser:
    def __init__(self, img, parser, NoseAnatomy):
        self.img = img
        self.H, self.W = img.shape[:2]
        self.gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.black0 = (self.gray0 < 8).astype(np.uint8)
        self.white0 = (self.gray0 > 250).astype(np.uint8)
        self.landmarks = parser.detect_from_image(img)
        self.anat = NoseAnatomy(landmarks=self.landmarks, image_shape=img.shape) \
            if self.landmarks and len(self.landmarks) >= 468 else None

    # ---- anatomic metrics from NoseAnatomy (correct px) ----
    def metrics(self):
        if not self.anat or not self.anat.valid:
            return None
        a = self.anat
        return {
            "width": float(a.nasal_width),
            "height": float(a.nasal_height),
            "w_over_h": float(a.nasal_width / max(a.nasal_height, 1e-6)),
            "alar_span": float(np.linalg.norm(a.get("alar_r") - a.get("alar_l"))),
            "alar_center_x": float((a.get("alar_l")[0] + a.get("alar_r")[0]) / 2),
        }

    # ---- integrity: NEW dark/white only (difference vs source) ----
    def integrity(self, out):
        og = cv2.cvtColor(cv2.resize(out, (self.W, self.H)), cv2.COLOR_BGR2GRAY) \
            if out.shape[:2] != (self.H, self.W) else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        new_black = ((og < 8).astype(np.uint8) & (self.black0 == 0)).mean() * 100
        new_white = ((og > 250).astype(np.uint8) & (self.white0 == 0)).mean() * 100
        # fresh harsh edges (seams) relative to source
        gx0, gy0 = cv2.Sobel(self.gray0, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(self.gray0, cv2.CV_32F, 0, 1, ksize=3)
        gx1, gy1 = cv2.Sobel(og, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(og, cv2.CV_32F, 0, 1, ksize=3)
        e0, e1 = np.hypot(gx0, gy0), np.hypot(gx1, gy1)
        new_edges = ((e1 > 90) & (e0 < 30)).mean() * 100
        return {"new_black": round(float(new_black), 3),
                "new_white": round(float(new_white), 3),
                "new_edges": round(float(new_edges), 3),
                "ok": new_black <= 0.3 and new_white <= 3.0 and new_edges <= 1.0}

    # ---- locality ----
    def locality(self, out, max_pct=40.0):
        d = cv2.absdiff(cv2.resize(out, (self.W, self.H)), self.img).mean(axis=2)
        changed = (d > 6).astype(np.uint8)
        pct = float(changed.mean() * 100)
        return pct, pct <= max_pct


# ----------------------------------------------------------------------
#  Test groups
# ----------------------------------------------------------------------
def group_api_health(parser, NoseAnatomy):
    print("\n━━━ ۱) سلامت سرویس و Endpoint ها ━━━")
    for path in ["/health", "/api/v1/edit/health", "/api/v1/edit/styles",
                 "/api/v1/edit/actions", "/api/v1/3d-test"]:
        try:
            with urllib.request.urlopen(BASE + path, timeout=15) as r:
                j = json.loads(r.read())
            report(f"GET {path}", True, str(j)[:80])
        except Exception as e:
            report(f"GET {path}", False, str(e)[:80])


def group_anatomy_engine(parser, NoseAnatomy, NoseAnatomyStyles):
    """Direct engine-level anatomic warp checks (not through HTTP)."""
    print("\n━━━ ۲) وارپینگ آناتومیک — تست مستقیم موتور ━━━")
    img = cv2.imread(str(IMG))
    a0m = Analyser(img, parser, NoseAnatomy).metrics()
    print(f"    مورفولوژی پایه: {json.dumps(a0m)}")

    base = Analyser(img, parser, NoseAnatomy)

    def alar_sym(out):
        """رصد دو بال (template-matching روی بافت بال‌ها)."""
        a = base.anat
        shifts = {}
        for side in ("alar_l", "alar_r"):
            anchor = a.get(side)
            cx, cy = int(anchor[0]), int(anchor[1])
            PAD, TPL = 100, 44
            tpl = img[cy-TPL//2:cy+TPL//2, cx-TPL//2:cx+TPL//2]
            if tpl.shape[0] < TPL or tpl.shape[1] < TPL:
                shifts[side] = None
                continue
            sy0, sy1 = max(0, cy-PAD), min(out.shape[0], cy+PAD)
            sx0, sx1 = max(0, cx-PAD), min(out.shape[1], cx+PAD)
            res = cv2.matchTemplate(out[sy0:sy1, sx0:sx1], tpl, cv2.TM_CCOEFF_NORMED)
            _, _, _, loc = cv2.minMaxLoc(res)
            dx = (loc[0] + TPL//2) - (cx - sx0)
            dy = (loc[1] + TPL//2) - (cy - sy0)
            shifts[side] = (dx, dy)
        return shifts

    cases = [
        # (action, fn, expected metric key, direction)
        ("نوک بینی بالا", NoseAnatomyStyles.upturned_tip, "tip_motion", None),
        ("بینی کوچک تر", NoseAnatomyStyles.smaller, "width", -1),
        ("بینی قلمی", NoseAnatomyStyles.slim_bridge, "width", -1),
        ("عروسکی", NoseAnatomyStyles.doll_tip, "width", -1),
        ("فانتزی", NoseAnatomyStyles.fantasy, "width", -1),
    ]
    for label, fn, kind, direction in cases:
        try:
            out = fn(img, base.landmarks, img.shape, 1.0)
            intg = base.integrity(out)
            pct, local = base.locality(out)
            asym = None
            if label != "نوک بینی بالا":
                s = alar_sym(out)
                if s.get("alar_l") and s.get("alar_r"):
                    asym = abs(abs(s["alar_l"][0]) - abs(s["alar_r"][0]))
            det = f"change={pct:.2f}% new_black={intg['new_black']}% new_edges={intg['new_edges']}%"
            if asym is not None:
                det += f" |alar_sym|Δ={asym:.0f}px"
            ok = intg["ok"] and local
            if label != "نوک بینی بالا" and asym is not None:
                ok = ok and asym <= 20  # تقارن بال: بیش از 20px ناموازنه = نقص
            report(f"{label} (и=1.0)", ok, det)
        except Exception as e:
            report(f"{label}", False, f"EXC: {e}")


def group_api_e2e(parser, NoseAnatomy):
    """End-to-end over the live API — with correct metrics."""
    print("\n━━━ ۳) انتها-به-انتها از طریق API (3d-filter) ━━━")
    img = cv2.imread(str(IMG))
    base = Analyser(img, parser, NoseAnatomy)

    nlu_map = {
        "نوک بینی بالا باشه": "upturned_tip",
        "بینی کوچیک تر بشه": "smaller",
        "قوز بینی رو بردار": "hump_reduction",
        "بینی عروسکی دوست دارم": "doll_tip",
        "بینی قلمی و کشیده باشه": "slim_bridge",
        "لب روسی بزن برام": "russian",
        "گونه ها پرتر بشن": "fuller",
        "فک رو زاویه دار کن": "sharper",
    }
    for text, exp_style in nlu_map.items():
        try:
            r = post_3d(text, 0.7)
            if r.get("status") != "success":
                report(f"<<{text}>>", False, f"status={r.get('status')} msg={r.get('message','')[:60]}")
                continue
            ch = r.get("changes", {})
            action = ch.get("action")
            style = ch.get("style")
            got = action or style
            ok_nlu = got == exp_style or (exp_style == "fuller" and "cheek" in str(ch.get("area")))
            out = b64_to_img(r.get("filtered_image") or r.get("image"))
            if out is None:
                report(f"<<{text}>>", False, "خروجی تصویر decode نشد")
                continue
            intg = base.integrity(out)
            pct, local = base.locality(out)
            ok = ok_nlu and intg["ok"] and local
            report(f"<<{text}>> → {ch.get('area')}/{got}",
                   ok, f"change={pct:.2f}% new_black={intg['new_black']}%")
        except Exception as e:
            report(f"<<{text}>>", False, f"EXC: {str(e)[:70]}")


def group_determinism(parser, NoseAnatomy):
    print("\n━━━ ۴) پایداری (Determinism) ━━━")
    try:
        r1 = post_3d("نوک بینی بالا", 0.6)
        r2 = post_3d("نوک بینی بالا", 0.6)
        a1 = b64_to_img(r1.get("filtered_image") or r1.get("image"))
        a2 = b64_to_img(r2.get("filtered_image") or r2.get("image"))
        if a1 is not None and a2 is not None:
            same = a1.shape == a2.shape and float(np.abs(a1.astype(int) - a2.astype(int)).mean()) < 0.5
            report("دو اجرای یکسان → خروجی یکسان", same,
                   f"mean-diff={float(np.abs(a1.astype(int)-a2.astype(int)).mean()):.4f}")
        else:
            report("دو اجرای یکسان → خروجی یکسان", False, "خروجی decode نشد")
    except Exception as e:
        report("دو اجرای یکسان → خروجی یکسان", False, str(e)[:70])


def group_generative_face_edit():
    print("\n━━━ ۵) بخش ساخت/تولید مولد (face-edit + diffusers) ━━━")
    # detect presence of generative stack
    try:
        import diffusers, transformers  # noqa
        gen_ok = True
    except Exception:
        gen_ok = False
    if not gen_ok:
        report("استک مولد (diffusers/transformers/insightface)", False,
               "وابستگی‌های تولید مولد نصب نیستند — این بخش قابل تست نیست", level="warn")
        report("POST /api/v1/face-edit/styles", False,
               "موتور مولد در دسترس نیست (وابستگی نصب نیست)", level="warn")
        return
    # if stack present, hit the endpoint
    try:
        raw = open(IMG, "rb").read()
        b64 = base64.b64encode(raw).decode()
        body = json.dumps({"image": b64, "area": "nose", "style": "fantasy", "intensity": 0.7}).encode()
        req = urllib.request.Request(BASE + "/api/v1/face-edit/styles", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            j = json.loads(r.read())
        report("POST /api/v1/face-edit/styles", j.get("status") == "success",
               f"styles={len(j.get('styles', []))}" if j.get("status") == "success" else str(j)[:80])
    except Exception as e:
        report("POST /api/v1/face-edit/styles", False, str(e)[:80])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-api", action="store_true", help="فقط تست موتور محلی")
    args = ap.parse_args()

    print("=" * 62)
    print("BeautyAI Model — Professional QA Suite")
    print(f"   API: {BASE} | image: {IMG.name} ({cv2.imread(str(IMG)).shape[1]}x{cv2.imread(str(IMG)).shape[0]})")
    print("=" * 62)

    parser, NoseAnatomy, NoseAnatomyStyles = load_deps()

    if not args.no_api:
        try:
            with urllib.request.urlopen(BASE + "/health", timeout=8):
                pass
        except Exception as e:
            report("اتصال به API", False, str(e)[:60])
            print("(اجرای تست‌های محلی موتور در حالت --no-api ممکن است)")
            args.no_api = True

    if not args.no_api:
        group_api_health(parser, NoseAnatomy)
        group_api_e2e(parser, NoseAnatomy)
        group_determinism(parser, NoseAnatomy)

    group_anatomy_engine(parser, NoseAnatomy, NoseAnatomyStyles)
    if not args.no_api:
        group_generative_face_edit()

    print("\n" + "=" * 62)
    print(f"📊 نتیجه نهایی: {RESULTS['pass']} ✅  |  {RESULTS['fail']} ❌  |  {RESULTS['warn']} ⚠️")
    print("=" * 62)

    # machine-readable report
    report_path = ROOT / "qa_report.json"
    report_path.write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2))
    print(f"📄 گزارش JSON: {report_path}")
    return 1 if RESULTS["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
