#!/bin/bash
# install_all.sh

echo "🚀 Installing BeautyAI Dependencies..."
echo "========================================"

# فعال کردن محیط مجازی
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found. Creating..."
    python -m venv venv
    source venv/bin/activate
fi

# ارتقای pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# نصب وابستگی‌ها
echo "📦 Installing requirements..."
pip install -r requirements.txt

# نصب MediaPipe با protobuf صحیح
echo "📦 Installing MediaPipe with correct dependencies..."
pip uninstall mediapipe protobuf -y
pip install protobuf==3.20.3
pip install mediapipe==0.10.8

# تست نصب
echo "🧪 Testing installation..."
python -c "
import mediapipe as mp
import torch
import fastapi
import cv2
print('✅ All packages installed successfully!')
print(f'✅ MediaPipe: {mp.__version__}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ FastAPI: {fastapi.__version__}')
print(f'✅ OpenCV: {cv2.__version__}')
"

echo "========================================"
echo "✅ Installation complete!"
echo "🔄 Run: source venv/bin/activate"