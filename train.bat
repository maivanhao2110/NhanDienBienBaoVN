@echo off
cd /d %~dp0
call yolov8_env\Scripts\activate
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=100 imgsz=640 batch=8 device=0
pause