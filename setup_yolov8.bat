@echo off
echo ================================
echo   SETUP YOLOv8 GPU (RTX 3060)
echo ================================

REM 👉 Di chuyen vao thu muc project (tu dong lay vi tri file bat)
cd /d %~dp0

echo.
echo [1] Tao virtual environment...
py -3.10 -m venv yolov8_env

echo.
echo [2] Activate environment...
call yolov8_env\Scripts\activate

echo.
echo [3] Update pip...
python -m pip install --upgrade pip

echo.
echo [4] Cai PyTorch CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [5] Cai YOLOv8...
pip install ultralytics

echo.
echo [6] Test GPU...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo [7] Train test YOLOv8 (10 epochs)...
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=10 imgsz=640

echo.
echo ================================
echo   DONE !!!
echo ================================
pause