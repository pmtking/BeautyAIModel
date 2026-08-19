# test.py
import mediapipe as mp

print("MediaPipe version:", mp.__version__)

# روش جدید برای import
try:
    import mediapipe.python.solutions as mp_solutions
    print("✅ Found solutions via mediapipe.python.solutions")
    from mediapipe.python.solutions import face_mesh
    print("✅ FaceMesh imported successfully")
except Exception as e:
    print("❌ Error with solutions:", e)

# روش قدیمی
try:
    face_mesh = mp.solutions.face_mesh
    print("✅ mp.solutions.face_mesh found")
except Exception as e:
    print("❌ mp.solutions.face_mesh error:", e)