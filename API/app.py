from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS
import easyocr

app = Flask(__name__)

# Khởi tạo thư mục static nếu chưa có
os.makedirs(os.path.join(os.getcwd(), 'static'), exist_ok=True)

# Tải model YOLOv8 (nếu chưa có best_v8.pt, sẽ dùng tạm yolov8n.pt để chạy demo)
model_path = 'best_v8.pt'
if not os.path.exists(model_path):
    print(f"Warning: Không tìm thấy {model_path}. Dùng tạm mô hình gốc yolov8n.pt.")
    model_path = 'yolov8n.pt'
model = YOLO(model_path)

print("Đang khởi tạo EasyOCR... (lần đầu sẽ tải model mất chút thời gian)")
reader = easyocr.Reader(['en'])
print("EasyOCR đã sẵn sàng!")

# Dictionary filter và mapping ý nghĩa biển báo dựa chuẩn xác trên 56 classes từ dataset.
# Lưu ý: Các class bên ngoài danh sách này (như người, xe, v.v) sẽ bị loại bỏ hoàn toàn.
sign_meanings = {
    # Báo hết lệnh cấm
    "DP-135": "Hết tất cả các lệnh cấm",
    
    # Nhóm biển cấm (P)
    "P-102": "Cấm đi ngược chiều",
    "P-103A": "Cấm ô tô",
    "P-103B": "Cấm ô tô rẽ phải",
    "P-103C": "Cấm ô tô rẽ trái",
    "P-104": "Cấm mô tô",
    "P-106A": "Cấm xe tải",
    "P-106B": "Cấm xe tải lớn",
    "P-107A": "Cấm ô tô khách và ô tô tải",
    "P-112": "Cấm người đi bộ",
    "P-115": "Hạn chế tải trọng",
    "P-117": "Hạn chế chiều cao",
    "P-123A": "Cấm rẽ trái",
    "P-123B": "Cấm rẽ phải",
    "P-124A": "Cấm quay đầu xe",
    "P-124B": "Cấm ô tô quay đầu xe",
    "P-124C": "Cấm rẽ trái và quay đầu xe",
    "P-127": "Hạn chế tốc độ tối đa",
    "P-128": "Cấm bấm còi",
    "P-130": "Cấm dừng và đỗ xe",
    "P-131A": "Cấm đỗ xe",
    "P-137": "Cấm rẽ trái và rẽ phải",
    "P-245A": "Đi chậm",
    
    # Nhóm biển Hiệu Lệnh & Chỉ dẫn (R)
    "R-301C": "Chỉ được rẽ trái",
    "R-301D": "Chỉ được rẽ phải",
    "R-301E": "Chỉ được rẽ trái", 
    "R-302A": "Vòng chướng ngại vật sang phải",
    "R-302B": "Vòng chướng ngại vật sang trái",
    "R-303": "Nơi giao nhau vòng xuyến",
    "R-407A": "Đường một chiều",
    "R-409": "Chỗ quay xe",
    "R-425": "Bệnh viện",
    "R-434": "Bến xe buýt",
    
    # Nhóm biển báo Phụ (S)
    "S-509A": "Chiều cao an toàn",
    
    # Nhóm biển cảnh báo Nguy Hiểm (W)
    "W-201A": "Chỗ ngoặt vòng bên trái",
    "W-201B": "Chỗ ngoặt vòng bên phải",
    "W-202A": "Nhiều chỗ ngoặt nguy hiểm",
    "W-202B": "Nhiều chỗ ngoặt nguy hiểm",
    "W-203B": "Đường hẹp bên trái",
    "W-203C": "Đường hẹp bên phải",
    "W-205A": "Đường giao nhau cùng cấp",
    "W-205B": "Đường giao nhau cùng cấp",
    "W-205D": "Đường giao nhau cùng cấp",
    "W-207A": "Giao nhau với đường không ưu tiên",
    "W-207B": "Giao nhau với đường không ưu tiên",
    "W-207C": "Giao nhau với đường không ưu tiên",
    "W-208": "Giao nhau với đường ưu tiên",
    "W-209": "Giao nhau có tín hiệu đèn",
    "W-210": "Giao nhau đường sắt có rào",
    "W-219": "Dốc xuống nguy hiểm",
    "W-224": "Đường người đi bộ cắt ngang",
    "W-225": "Trẻ em",
    "W-227": "Công trường",
    "W-233": "Nguy hiểm khác",
    "W-235": "Đường đôi",
    "W-245A": "Đi chậm"
}

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def process_image():
    if 'image_name' not in request.files:
        return jsonify({'error': 'No file uploaded'})
        
    file = request.files['image_name']
    img_path = os.path.join(os.getcwd(), 'static', 'upload_image.jpg')
    file.save(img_path)

    # Xử lý qua YOLOv8
    results = model.predict(source=img_path)
    result = results[0]
    
    result_img_path = os.path.join(os.getcwd(), 'static', 'result.jpg')
    result.save(filename=result_img_path)
    
    # Trích xuất class (kèm filter)
    detected_names_set = set()
    orig_img = result.orig_img
    
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id].upper()
        
        # Chỉ nhận diện nếu có trong dictionary
        if class_name in sign_meanings:
            meaning = sign_meanings[class_name]
            
            # Module nhận diện chữ số OCR cho tốc độ
            if class_name == "P-127":
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                crop_img = orig_img[int(y1):int(y2), int(x1):int(x2)]
                
                # Cắt và đọc
                ocr_results = reader.readtext(crop_img, allowlist='0123456789')
                best_text = ""
                highest_prob = 0
                for label_bbox, text, prob in ocr_results:
                    if text.isdigit() and prob > highest_prob:
                        best_text = text
                        highest_prob = prob
                        
                if best_text:
                    meaning = f"{meaning} {best_text} km/h"
                    
            detected_names_set.add(meaning)
    
    detected_names = list(detected_names_set)

    if len(detected_names) > 0:
        text_to_speak = "Phát hiện: " + ", ".join(detected_names)
    else:
        text_to_speak = "Không phát hiện thấy biển báo nào"
        
    # Gọi gTTS sinh file mp3
    tts = gTTS(text=text_to_speak, lang='vi')
    audio_path = os.path.join(os.getcwd(), 'static', 'voice.mp3')
    if os.path.exists(audio_path):
        os.remove(audio_path)
    tts.save(audio_path)

    timestamp = int(time.time())
    return jsonify({
        'result_url': f"/static/result.jpg?t={timestamp}",
        'audio_url': f"/static/voice.mp3?t={timestamp}",
        'detected_names': detected_names
    })

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data payload'}), 400
        
    # Lấy các tham số cấu hình từ frontend
    use_roi = data.get('use_roi', False)
    use_distance = data.get('use_distance', False)
    car_only = data.get('car_only', False)
    min_height = data.get('min_height', 25)
    
    base64_img = data['image']
    if "," in base64_img:
        base64_img = base64_img.split(",")[1]
        
    img_data = base64.b64decode(base64_img)
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image data'}), 400

    results = model.predict(source=img, verbose=False)
    result = results[0]
    
    detections = []
    detected_names_set = set()
    orig_img = result.orig_img
    h_frame, w_frame = img.shape[:2]
    
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = model.names[class_id].upper()
        
        # Lọc bỏ tất cả đối tượng không thuộc biển báo
        if class_name not in sign_meanings:
            continue
            
        # Lọc bỏ biển báo KHÔNG liên quan tới xe ô tô con/ô tô cá nhân
        if car_only:
            # Danh sách các mã biển báo: Không ảnh hưởng, Cấm xe tải, Cấm mô tô, Cấm đi bộ, Bến xe buýt...
            non_car_signs = ["P-104", "P-106A", "P-106B", "P-107A", "P-112", "R-434"]
            if class_name in non_car_signs:
                continue

        meaning = sign_meanings[class_name]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # 1. Lọc theo ROI (Chỉ giữ biển báo bên phải)
        x_center = (x1 + x2) / 2
        if use_roi:
            if x_center <= 0.3 * w_frame:
                continue # Loại bỏ
            
            # Logic ưu tiên (gửi về frontend để vẽ màu khác nhau nếu cần)
            priority = "high" if x_center > 0.6 * w_frame else "medium"
        else:
            priority = "none"

        # 2. Lọc theo khoảng cách (dựa trên chiều cao bbox)
        bbox_height = y2 - y1
        if use_distance and bbox_height < min_height:
            continue # Loại bỏ biển ở xa (>40m)

        # Module nhận diện chữ số OCR cho tốc độ
        if class_name == "P-127":
            crop_img = orig_img[int(y1):int(y2), int(x1):int(x2)]
            
            # Cắt và đọc
            ocr_results = reader.readtext(crop_img, allowlist='0123456789')
            best_text = ""
            highest_prob = 0
            for label_bbox, text, prob in ocr_results:
                if text.isdigit() and prob > highest_prob:
                    best_text = text
                    highest_prob = prob
                    
            if best_text:
                meaning = f"{meaning} {best_text} km/h"
                
        detections.append({
            'box': [x1, y1, x2, y2],
            'class_name': class_name,
            'meaning': meaning,
            'confidence': conf,
            'priority': priority,
            'height': bbox_height
        })
        detected_names_set.add(meaning)
        
    return jsonify({
        'detections': detections,
        'detected_classes': list(detected_names_set)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)