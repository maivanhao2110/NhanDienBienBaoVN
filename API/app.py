from flask import Flask, render_template, request, jsonify, url_for
import os
import time
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from gtts import gTTS

app = Flask(__name__)

# Khởi tạo thư mục static nếu chưa có
os.makedirs(os.path.join(os.getcwd(), 'static'), exist_ok=True)

# Tải model YOLOv8 (nếu chưa có best_v8.pt, sẽ dùng tạm yolov8n.pt để chạy demo)
model_path = 'best_v8.pt'
if not os.path.exists(model_path):
    print(f"Warning: Không tìm thấy {model_path}. Dùng tạm mô hình gốc yolov8n.pt.")
    model_path = 'yolov8n.pt'
model = YOLO(model_path)

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
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id].upper()
        
        # Chỉ nhận diện nếu có trong dictionary
        if class_name in sign_meanings:
            detected_names_set.add(sign_meanings[class_name])
    
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
    
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        class_name = model.names[class_id].upper()
        
        # Lọc bỏ tất cả đối tượng không thuộc biển báo
        if class_name not in sign_meanings:
            continue
            
        meaning = sign_meanings[class_name]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        detections.append({
            'box': [x1, y1, x2, y2],
            'class_name': class_name,
            'meaning': meaning,
            'confidence': conf
        })
        detected_names_set.add(meaning)
        
    return jsonify({
        'detections': detections,
        'detected_classes': list(detected_names_set)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)