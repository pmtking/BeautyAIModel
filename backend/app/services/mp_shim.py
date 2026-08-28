# app/services/mp_shim.py
"""
سازگارکننده MediaPipe برای API جدید (Tasks) — جایگزین mp.solutions
=================================================================
در mediapipe >= 0.10.30 ساخته‌شده برای پایتون 3.12+، ماژول قدیمی
`mp.solutions.face_mesh` حذف شده و فقط Tasks API باقی مانده.

این شیم یک کلاس FaceMesh با همان اینترفیس قدیمی می‌سازد:
    fm = mp_shim.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    refine_landmarks=True, min_detection_confidence=0.5)
    res = fm.process(rgb_image)   # → {multi_face_landmarks: [ ... ]}
    fm.close()

خروجی process دقیقاً مثل قبل است: هر landmark دارای x,y,z (نرمال 0..1).
اگر mediapipe قدیمی (با solutions) نصب باشد، شیم خودکار به همان برگردانده می‌شود.
"""
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

_MODEL_CANDIDATES = [
    "/opt/buti/backend_full/app/services/face_landmarker.task",
    "/opt/buti/backend_full/face_landmarker.task",
    "/opt/buti/backend/app/services/face_landmarker.task",
    "/tmp/face_landmarker.task",
    os.path.expanduser("~/face_landmarker.task"),
]


class _Landmark:
    """شبیه‌سازی landmark.solutions — فقط x,y,z"""

    __slots__ = ("x", "y", "z", "visibility", "presence")

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=None, presence=None):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class _NormalizedLandmarkList:
    """شبیه‌سازی res.multi_face_landmarks[0]"""

    def __init__(self, landmarks: List[_Landmark]):
        self.landmark = landmarks


class _FaceMeshResult:
    def __init__(self, landmarks: Optional[List[_Landmark]]):
        self.multi_face_landmarks = (
            [_NormalizedLandmarkList(landmarks)] if landmarks else None
        )


class _FaceMesh:
    """کلاس سازگار با اینترفیس قدیمی mp.solutions.face_mesh.FaceMesh"""

    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        **kwargs,
    ):
        self._landmarker = None
        self._refine = refine_landmarks
        self._conf = min_detection_confidence
        self._max_faces = max_num_faces
        try:
            self._landmarker = self._build()
            logger.info("MP Tasks FaceLandmarker initialized")
        except Exception as e:
            logger.error(f"FaceLandmarker init failed: {e}")
            self._landmarker = None

    # ---------------------------------------------------------
    def _find_model(self) -> Optional[str]:
        for p in _MODEL_CANDIDATES:
            if os.path.exists(p):
                return p
        return None

    def _build(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        model_path = self._find_model()
        if not model_path:
            raise FileNotFoundError(
                "face_landmarker.task not found in " + ", ".join(_MODEL_CANDIDATES)
            )
        opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=self._max_faces,
            min_face_detection_confidence=self._conf,
            min_face_presence_confidence=self._conf,
            min_tracking_confidence=self._conf,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    # ---------------------------------------------------------
    def process(self, rgb_image):
        """rgb_image: np.ndarray (H,W,3) uint8 → همان شکل خروجی قدیمی"""
        if self._landmarker is None:
            return _FaceMeshResult(None)
        try:
            from mediapipe.tasks.python import vision

            mp_image = vision.Image(
                image_format=vision.ImageFormat.SRGB, data=rgb_image
            )
            res = self._landmarker.detect(mp_image)
        except Exception as e:
            logger.warning(f"FaceLandmarker detect error: {e}")
            return _FaceMeshResult(None)

        if not res.face_landmarks:
            return _FaceMeshResult(None)

        lms = []
        for lm in res.face_landmarks[0]:
            lms.append(
                _Landmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(lm.z),
                    visibility=getattr(lm, "visibility", None),
                    presence=getattr(lm, "presence", None),
                )
            )
        return _FaceMeshResult(lms)

    def close(self):
        if self._landmarker is not None:
            try:
                self._landmarker.close()
            except Exception:
                pass
            self._landmarker = None


class _FaceMeshModule:
    """شبیه‌سازی mp.solutions.face_mesh"""

    FaceMesh = _FaceMesh
    FACEMESH_CONTOURS = None
    FACEMESH_TESSELATION = None


# اگر mediapipe قدیمی با solutions هست، از همان استفاده کن
def _get_face_mesh_module():
    try:
        import mediapipe as mp

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            return mp.solutions.face_mesh
    except Exception:
        pass
    return _FaceMeshModule()


face_mesh = _get_face_mesh_module()