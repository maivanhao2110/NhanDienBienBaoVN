from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# 1. Load 2 model (đảm bảo đường dẫn file .pt chính xác)
model_v8 = YOLO('yolov8_best.pt')
# Lưu ý: Với v7, nếu chưa cài thư viện riêng, bạn có thể dùng tạm ảnh detect từ Kaggle
# Hoặc nếu đã có ultralytics, v8 cũng có thể load một số bản v7 đã convert

# 2. Chạy dự đoán trên cùng 1 ảnh
img_path = 'duong/dan/den/anh_test.jpg' 
res_v8 = model_v8(img_path)[0].plot()

# 3. Hiển thị và lưu
cv2.imshow('Ket qua YOLOv8', res_v8)
cv2.imwrite('so_sanh_v8.jpg', res_v8)
cv2.waitKey(0)