"""مانورهای v3 نهایی — با تکنیک vertical-stretch که اثبات شد کار میکند
نتایج تست:
  - upturn با stretch: diff=7.68 ✅ (قوی!)
  - tip_small radial:  diff=0.42 ⚠️ (باید قوی‌تر شود)
  - nostril shrink:    diff=0.45 ✅
"""
import sys, cv2, numpy as np
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src')
sys.path.insert(0, '/home/mohammad/Desktop/project/BeautyAIModel/ai_training/src/models')
from face_parser.model import FaceParserModel
from warping.nose_anatomy import NoseAnatomy
from warping.nose_styles import _roi_bounds


class V3Maneuvers:
    def __init__(self):
        self.parser = FaceParserModel()

    def detect(self, img):
        lm = self.parser.detect_from_image(img)
        return NoseAnatomy(landmarks=lm, image_shape=img.shape) if lm else None

    # ═══════ نوک سر بالا — stretch عمودی ═══════
    def upturn(self, img, anat, intensity=0.7):
        tip = anat.get('tip'); radix = anat.get('radix')
        al_l, al_r = anat.get('alar_l'), anat.get('alar_r')
        w = anat.nasal_width
        pts = anat.ordered_array()
        if pts is None:
            return img.copy()
        x0,y0,x1,y1 = _roi_bounds(img.shape, pts, pad_frac=0.6)
        ys,xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)
        axis_x = (al_l[0]+al_r[0])/2
        y_alar = float((al_l[1]+al_r[1])/2)
        prog = np.clip((ys-y_alar)/(tip[1]-y_alar), 0, 1.4)
        vert = prog**1.8
        lat = np.clip(1-np.abs(xs-axis_x)/(w*0.75),0,1)**1.3
        weight = vert*lat*float(intensity)
        map_x = (xs-x0).astype(np.float32)
        map_y = (ys+weight*w*0.22-y0).astype(np.float32)
        warped = cv2.remap(img[y0:y1,x0:x1],map_x,map_y,
                           cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        out = img.copy()
        m3 = weight[...,None]
        out[y0:y1,x0:x1] = (warped*m3+out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)
        # سایه زیر نوک جدید برای عمق
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        new_tip_y = tip[1] - w*0.22*intensity
        d = np.abs(ys-new_tip_y)
        shadow = np.clip(1-d/(w*0.10),0,1)*lat
        lab[...,0][y0:y1,x0:x1] += shadow*12
        lab[...,0] = np.clip(lab[...,0],0,255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ═══════ قوز — فشار عمودی روی زون قوز + روشن‌سازی ═══════
    def hump(self, img, anat, intensity=0.7):
        radix = anat.get('radix'); tip = anat.get('tip')
        w = anat.nasal_width; h = anat.nasal_height
        pts = anat.ordered_array()
        if pts is None:
            return img.copy()
        x0,y0,x1,y1 = _roi_bounds(img.shape, pts, pad_frac=0.6)
        ys,xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)
        u = tip-radix
        u_len = float(np.linalg.norm(u))+1e-6
        u_n = u/u_len
        n_v = np.array([-u_n[1],u_n[0]],dtype=np.float32)
        rel_x = xs-radix[0]; rel_y = ys-radix[1]
        along = (rel_x*u_n[0]+rel_y*u_n[1])/u_len
        perp  =  rel_x*n_v[0]+rel_y*n_v[1]
        zone = np.clip(1-np.abs(along-0.48)/0.24,0,1)**1.4
        band = np.clip(1-np.abs(perp)/(w*0.30),0,1)**1.2
        weight = zone*band*float(intensity)
        dy = weight*h*0.055
        map_x = (xs-x0).astype(np.float32)
        map_y = (ys+dy-y0).astype(np.float32)
        warped = cv2.remap(img[y0:y1,x0:x1],map_x,map_y,
                           cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        out = img.copy()
        m3 = weight[...,None]*0.9
        out[y0:y1,x0:x1] = (warped*m3+out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[...,0][y0:y1,x0:x1] += weight*16
        lab[...,0] = np.clip(lab[...,0],0,255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ═══════ کوچک‌سازی نوک — radial قوی‌تر ═══════
    def tip_small(self, img, anat, intensity=0.7):
        tip = anat.get('tip')
        w = anat.nasal_width
        pts = anat.ordered_array()
        if pts is None:
            return img.copy()
        x0,y0,x1,y1 = _roi_bounds(img.shape, pts, pad_frac=0.5)
        ys,xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)
        radius = w*0.48
        d = np.sqrt((xs-tip[0])**2+(ys-tip[1])**2)
        wt = np.clip(1-d/radius,0,1)**1.3
        scale = 1 - float(intensity)*0.38*wt
        map_x = (tip[0]+(xs-tip[0])*scale-x0).astype(np.float32)
        map_y = (tip[1]+(ys-tip[1])*scale-y0).astype(np.float32)
        warped = cv2.remap(img[y0:y1,x0:x1],map_x,map_y,
                           cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
        out = img.copy()
        m3 = wt[...,None]*0.95
        out[y0:y1,x0:x1] = (warped*m3+out[y0:y1,x0:x1]*(1-m3)).astype(np.uint8)
        return out

    # ═══════ تنگ‌کردن سوراخ‌ها — shrink حلقه + سایه داخلی ═══════
    def nostrils(self, img, anat, intensity=0.6):
        nl, nr = anat.get('nostril_l'), anat.get('nostril_r')
        w = anat.nasal_width
        pts = anat.ordered_array()
        if pts is None or nl is None or nr is None:
            return img.copy()
        x0,y0,x1,y1 = _roi_bounds(img.shape, pts, pad_frac=0.4)
        ys,xs = np.mgrid[y0:y1,x0:x1].astype(np.float32)

        result = img.copy()
        for c in (nl, nr):
            radius = w*0.26
            d = np.sqrt((xs-c[0])**2+(ys-c[1])**2)
            wt = np.clip(1-d/radius,0,1)**1.5
            scale = 1 - float(intensity)*0.42*wt
            roi = result[y0:y1,x0:x1]
            map_x = (c[0]+(xs-c[0])*scale-x0).astype(np.float32)
            map_y = (c[1]+(ys-c[1])*scale-y0).astype(np.float32)
            warped = cv2.remap(roi,map_x,map_y,
                               cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT)
            m3 = wt[...,None]*0.94
            result[y0:y1,x0:x1] = (warped*m3+roi*(1-m3)).astype(np.uint8)

        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        for c in (nl, nr):
            d = np.sqrt((xs-c[0])**2+(ys-c[1])**2)
            inner = np.clip(1-d/(w*0.09),0,1)**1.2
            lab[...,0][y0:y1,x0:x1] -= inner*14*float(intensity)
        lab[...,0] = np.clip(lab[...,0],0,255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
