from ultralytics import YOLO

# Khởi tạo model từ pretrained model của YOLOv8
model = YOLO('yolov8n.pt')  # Có thể đổi thành yolov8s.pt, yolov8m.pt tùy cấu hình máy

# Bắt đầu quá trình huấn luyện
# Lưu ý: Cần chỉnh 'data_custom.yaml' trỏ đến đúng thư mục dataset của bạn
results = model.train(
    data='data_custom.yaml', # Đường dẫn đến file yaml cấu hình class và thư mục dataset
    epochs=100,              # Số epoch
    imgsz=640,               # Kích thước ảnh đầu vào
    batch=16,                # Batch size
    name='traffic_sign_v8'   # Tên thư mục lưu kết quả ở mục runs/detect/
)

# Kết quả sau khi train xong sẽ có một file best.pt nằm ở runs/detect/traffic_sign_v8/weights/
# Hãy copy file best.pt đó vào thư mục API để chạy ứng dụng giao diện.
