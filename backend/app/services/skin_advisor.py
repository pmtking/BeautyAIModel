"""
🧴 مشاور پوست BeautyAI — تحلیل + روتین شخصی‌سازی‌شده
======================================================
بر اساس Baumann Skin Typing System (۱۶ نوع) + تشخیص تصویری
فاز ۱: بدون GPU — قواعد کلینیکی + تحلیل رنگ/بافت OpenCV
"""
import cv2
import numpy as np
from typing import Dict, List, Optional


# ═══════════════════════════════════════
#  ۱. تایپ بامان از پرسشنامه
# ═══════════════════════════════════════
QUIZ = [
    # (key, question, option_O_or_S_or_P_or_W, option_opposite)
    {"key": "oil",    "q": "چند ساعت بعد از شستشو، پوستم برق میزند؟",
     "opts": [("هرگز — کشیده می‌شود", "D"), ("فقط پیشانی/بینی", "M"),
              ("کل صورت تا ظهر", "O")]},
    {"key": "acne",   "q": "جوش و آکنه مکرر دارم؟",
     "opts": [("تقریباً هیچ‌وقت", "R"), ("گاهی", "S"), ("مداوم", "S")]},
    {"key": "sens",   "q": "با محصولات جدید سوزش/قرمزی میگیرم؟",
     "opts": [("هرگز", "R"), ("بعضی وقتا", "S"), ("همیشه", "S")]},
    {"key": "spots",  "q": "لک آفتاب یا حاملگی دارم؟",
     "opts": [("نه", "N"), ("کم", "P"), ("زیاد", "P")]},
    {"key": "wrinkle","q": "خطوط ریز دور چشم/پیشانی دائم دیدم؟",
     "opts": [("نه هنوز", "T"), ("کم‌کم داره شروع میشه", "W"), ("بله واضح", "W")]},
]


def baumann_type(answers: Dict[str, str]) -> str:
    """ترکیب ۴ حرف: O/D + S/R + P/N + W/T"""
    o = answers.get("oil", "O")           # O/D/M→D
    s = answers.get("sens", "R") or answers.get("acne", "R")
    p = answers.get("spots", "N")
    w = answers.get("wrinkle", "T")
    if o == "M":
        o = "O"                            # ترکیبی → چرب محسوب
    return f"{o}{s}{p}{w}"


# ═══════════════════════════════════════
#  ۲. تحلیل تصویر — متریک‌های OpenCV
# ═══════════════════════════════════════
class SkinImageAnalyzer:

    def analyze(self, image_bgr) -> Dict:
        """تحلیل کامل از یک سلفی — خروجی متریک‌ها + مشکلات"""
        img = cv2.resize(image_bgr, (512, 512))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, A, B = lab[...,0], lab[...,1], lab[...,2]

        h, w_ = L.shape
        cheeks = [(int(h * 0.55), int(w_ * 0.28)), (int(h * 0.55), int(w_ * 0.72))]
        forehead = (int(h*.25), w_//2)

        metrics = {}
        metrics["oiliness"]   = self._shine(L)
        metrics["redness"]    = self._erythema(A)
        metrics["pores"]      = self._pores(img, cheeks[0])
        metrics["texture"]    = self._texture(L, cheeks)
        metrics["spots"]      = self._spots(L, A, cheeks)
        metrics["dark_circles"] = self._dark_circles(L, h, w_)
        metrics["acne_like"]  = self._acne_blobs(A, cheeks)

        issues = self._interpret(metrics)
        skin_axis = self._guess_od(metrics)
        return {"metrics": {k: round(float(v),3) for k,v in metrics.items()},
                "issues": issues,
                "od_guess": skin_axis}

    def _shine(self, L):
        """براقی = درصد پیکسل‌های خیلی روشن در T-zone تقریبی"""
        tz = L[int(L.shape[0]*.18):int(L.shape[0]*.62), :]
        return float((tz > 232).mean())

    def _erythema(self, A):
        """قرمزی = میانگین کانال a در گونه‌ها بالا"""
        h, w_ = A.shape
        roi = A[int(h*.45):int(h*.68), int(w_*.15):int(w_*.85)]
        return float(roi.mean()/255*2 - 1)

    def _pores(self, bgr, center):
        """منافذ = تعداد نقاط تیره کوچک با local-threshold"""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        y, x = center
        r = 60
        roi = gray[max(0,y-r):y+r, max(0,x-r):x+r]
        if roi.size < 100: return 0
        blur = cv2.medianBlur(roi, 5)
        dark = cv2.threshold(blur, 0, 255,
                             cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        cnts,_ = cv2.findContours(dark, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        small = [c for c in cnts if 2 < cv2.contourArea(c) < 60]
        return len(small)/100.0

    def _texture(self, L, cheeks):
        """بافت = واریانس Laplacian (کم = صاف، زیاد = ناهموار)"""
        vals = []
        for (y,x) in cheeks:
            r=50; roi=L[max(0,y-r):y+r,max(0,x-r):x+r]
            vals.append(cv2.Laplacian(roi,cv2.CV_32F).var())
        return float(np.mean(vals))/500

    def _spots(self, L, A, cheeks):
        """لک = blobهای تیره‌تر از اطراف با اندازه متوسط"""
        blur = cv2.GaussianBlur(L,(0,0),9)
        diff = blur - L                      # تیره‌تر از اطراف
        mask = (diff>8).astype(np.uint8)
        n,_ ,stats,_ = cv2.connectedComponentsWithStats(mask)
        medium = sum(1 for i in range(1,n)
                     if 30 < stats[i,cv2.CC_STAT_AREA] < 1500)
        return medium/50.0

    def _dark_circles(self, L, h_, w_):
        eye_l = L[int(h_*0.42):int(h_*0.52), int(w_*0.12):int(w_*0.38)]
        cheek = L[int(h_*0.58):int(h_*0.66), int(w_*0.12):int(w_*0.38)]
        if eye_l.size==0 or cheek.size==0: return 0
        return float(max(0, (cheek.mean()-eye_l.mean())/60))

    def _acne_blobs(self, A, cheeks):
        """قرمزی‌های محلی گرد = احتمال جوش فعال"""
        out = 0
        for (y,x) in cheeks:
            r=70; a=A[max(0,y-r):y+r,max(0,x-r):x+r]
            if a.size<200: continue
            amax = cv2.dilate(a, np.ones((9,9),np.uint8))
            peaks = ((a>=amax)&(a>np.percentile(a,92))).astype(np.uint8)
            n,_ ,st,_ = cv2.connectedComponentsWithStats(peaks)
            out += sum(1 for i in range(1,n)
                       if 15<st[i,cv2.CC_STAT_AREA]<300)/20
        return float(out)

    def _interpret(self, m) -> List[Dict]:
        issues=[]
        if m["acne_like"]>1.5:  issues.append({"type":"acne","confidence":min(.95,.5+m["acne_like"]/6),"region":"cheeks"})
        if m["spots"]>1.2:      issues.append({"type":"dark_spots","confidence":min(.9,.4+m["spots"]/5),"region":"cheeks"})
        if m["pores"]>1.5:      issues.append({"type":"large_pores","confidence":min(.85,.4+m["pores"]/6),"region":"nose_cheeks"})
        if m["dark_circles"]>.35:issues.append({"type":"dark_circles","confidence":min(.88,.4+m["dark_circles"]),"region":"under_eye"})
        if m["redness"]>.30:    issues.append({"type":"redness","confidence":min(.8,.4+m["redness"]),"region":"cheeks"})
        if m["oiliness"]>.06:   issues.append({"type":"oily_skin","confidence":min(.9,.5+m["oiliness"]/0.2),"region":"t_zone"})
        return issues

    def _guess_od(self, m) -> str:
        return "O" if (m["oiliness"]>0.05 or m["pores"]>1.6) else "D"


# ═══════════════════════════════════════
#  ۳. سازنده روتین
# ═══════════════════════════════════════
class RoutineBuilder:

    SCIENCE = {
        "cleanser": "پاک‌کننده pH-متوازن، بدون سولفات",
        "vitamin_c": "ویتامین C 10-15% — آنتی‌اکسیدان و ضدلک (فقط صبح)",
        "niacinamide": "نیاسینامید ۵% — منافذ و چربی؛ با همه سازگار",
        "hyaluronic": "هیالورونیک اسید — آبرسان عمقی",
        "retinol": "رتینول ۰.۰۲۵% شروع؛ فقط شب؛ هفته‌ای ۲ بار شروع",
        "aha_bha": "BHA سالیسیلیک برای منافذ/جوش — یک شب در میان",
        "azelaic": "آزلائیک اسید ۱۰% — قرمزی و لک، بارداری هم امن",
        "spf": "ضدآفتاب SPF50+ PA++++ — مهم‌ترین مرحله ضدپیری",
        "ceramide": "مرطوب‌کننده سرامید — ترمیم سد پوستی",
    }

    def build(self, btype: str, issues: List[Dict], age: int=30) -> Dict:
        O = btype.startswith("O"); S = "S" in btype[1]
        P = "P" in btype[:3]; W = "W" in btype
        types = {i["type"] for i in issues}

        morning, evening, notes = [], [], []

        # ── پاک‌کننده ──
        morning.append({"step":"cleanser","product":"فوم ملایم" if O else "شیر پاک‌کننده",
                        "why":self.SCIENCE["cleanser"]})
        evening.append(morning[-1].copy())

        # ── تونر/درمان ──
        if "large_pores" in types or O:
            eve_t = {"step":"bha","product":"BHA 2% (یک شب در میان)",
                     "why":self.SCIENCE["aha_bha"]}
            evening.append(eve_t); notes.append("BHA را یک شب در میان استفاده کن")

        # ── سرم ──
        if P or "dark_spots" in types:
            morning.append({"step":"vitamin_c","product":"سرم ویتامین C",
                            "why":self.SCIENCE["vitamin_c"]})
        if "acne" in types or O:
            morning.append({"step":"niacinamide","product":"نیاسینامید ۵%",
                            "why":self.SCIENCE["niacinamide"]})
        if "redness" in types or S:
            evening.append({"step":"azelaic","product":"آزلائیک ۱۰%",
                            "why":self.SCIENCE["azelaic"]})

        # ── مرطوب‌کننده ──
        moist = "ژل سبک" if O else "کرم غنی"
        morning.append({"step":"moisturizer","product":moist,
                        "why":self.SCIENCE["ceramide"]})
        evening.append({"step":"moisturizer","product":f"{moist} ترمیمی",
                        "why":self.SCIENCE["ceramide"]})

        # ── رتینول (شب، اگر W و نه حساس شدید) ──
        if (W or age >= 27) and not (S and "acne" in types):
            evening.insert(-1, {"step":"retinol","product":"رتینول شروع‌کننده",
                                "why":self.SCIENCE["retinol"]})
            notes.append("رتینول را فقط شب، هفته‌ای ۲ بار شروع کن و کم‌کم زیاد کن")

        # ── SPF ──
        morning.append({"step":"sunscreen","product":"SPF50+ PA++++",
                        "why":self.SCIENCE["spf"]})
        notes.append("بدون ضدآفتاب، همه درمان‌های لک بی‌اثر است!")

        if "dark_circles" in types:
            notes.append("تیرگی زیر چشم: خواب کافی + سرم کافئین/ویت K")
        if "acne" in types:
            notes.append("جوش را دست نزن؛ تغییر روتین هر ۴ هفته ارزیابی شود")
        return {
            "baumann_type": btype,
            "routine": {"morning": morning, "evening": evening},
            "notes": notes,
        }


def advise(image_bgr, quiz_answers: Optional[dict]=None, age:int=30):
    """API اصلی — تصویر + پرسشنامه → تحلیل + روتین"""
    analyzer = SkinImageAnalyzer()
    result_img = analyzer.analyze(image_bgr)
    answers = dict(quiz_answers or {})
    # ادغام حدس تصویری با پرسشنامه
    answers.setdefault("oil", result_img["od_guess"])
    btype = baumann_type(answers)
    rb = RoutineBuilder().build(btype, result_img["issues"], age)
    rb["image_metrics"] = result_img["metrics"]
    rb["detected_issues"] = result_img["issues"]
    return rb
