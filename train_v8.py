from ultralytics import YOLO
import torch

def train_with_gpu():
    # Kiểm tra xem máy đã nhận GPU chưa
    if torch.cuda.is_available():
        device = 0
        print(f"--- Tuyệt vời! Đang chạy bằng GPU: {torch.cuda.get_device_name(0)} ---")
    else:
        device = 'cpu'
        print("--- Cảnh báo: Vẫn đang dùng CPU. Kiểm tra lại Bước 2.1 ---")

    model = YOLO('yolov8n.pt') 
    yaml_path = r'D:/Study_now/TriTueNhanTao/NhanDienBienBaoVN/yolov7/data/bienbao.yaml'

    model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=16,        # Giảm xuống 16 cho RTX 3050
        device=0,
        workers=4,       # Để 0 trên Windows để tránh lỗi đa luồng
        amp=False,       # TẮT AMP ĐỂ SỬA LỖI 'unable to find an engine'
        project='DoAn_SoSanh', 
        name='YOLOv8_GPU_Run',
        plots=True
    )

if __name__ == '__main__':
    train_with_gpu()