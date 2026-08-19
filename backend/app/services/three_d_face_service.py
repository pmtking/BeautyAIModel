import cv2
import numpy as np
import base64
from typing import Dict, Optional, List
import logging
import mediapipe as mp

logger = logging.getLogger(__name__)


class ThreeDFaceService:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        logger.info("✅ ThreeDFaceService initialized")

    def create_3d_texture(self, image: np.ndarray, modifications: Dict) -> Dict:
        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                return {'error': 'چهره‌ای شناسایی نشد'}

            landmarks_3d = []
            for lm in result.multi_face_landmarks[0].landmark:
                landmarks_3d.append({'x': lm.x, 'y': lm.y, 'z': lm.z})

            vertices = self._create_vertices(landmarks_3d)
            faces = self._create_faces()
            modified_vertices = self._apply_modifications(vertices, modifications)

            _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            texture_base64 = base64.b64encode(buffer).decode('utf-8')

            return {
                'status': 'success',
                'vertices': modified_vertices.tolist(),
                'faces': faces,
                'texture': texture_base64,
                'num_vertices': len(modified_vertices),
                'num_faces': len(faces)
            }

        except Exception as e:
            logger.error(f"Error: {e}")
            return {'error': str(e)}

    def _create_vertices(self, landmarks_3d: List[Dict]) -> np.ndarray:
        vertices = []
        for lm in landmarks_3d:
            vertices.append([lm['x'] * 2 - 1, (1 - lm['y']) * 2 - 1, lm['z'] * 2])
        return np.array(vertices, dtype=np.float32)

    def _create_faces(self) -> List[List[int]]:
        faces = []
        indices = list(range(0, 468, 3))
        for i in range(len(indices) - 2):
            faces.append([indices[i], indices[i + 1], indices[i + 2]])
        return faces

    def _apply_modifications(self, vertices: np.ndarray, modifications: Dict) -> np.ndarray:
        modified = vertices.copy()
        area = modifications.get('area')
        intensity = modifications.get('intensity', 0.5)

        if not area:
            return modified

        area_indices = {
            'nose': list(range(1, 36)),
            'lip': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308],
            'jaw': list(range(0, 17)),
            'cheek': list(range(123, 135)) + list(range(135, 147)),
            'eye': list(range(33, 46)) + list(range(362, 374))
        }

        indices = area_indices.get(area, [])

        if area == 'nose':
            factor = 1 - (intensity * 0.15)
            for idx in indices:
                if idx < len(modified):
                    modified[idx, 0] = modified[idx, 0] * factor

        elif area == 'lip':
            factor = 1 + (intensity * 0.15)
            for idx in indices:
                if idx < len(modified):
                    modified[idx, 1] = modified[idx, 1] * factor

        return modified


three_d_service = ThreeDFaceService()