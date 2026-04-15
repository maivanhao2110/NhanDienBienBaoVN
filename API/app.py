from flask import Flask, render_template, request, url_for
import os
import shutil
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

# Thử chuyển đổi một số ID sang tiếng Việt nếu dùng mô hình mặc định (COCO)
# Giả sử mô hình của bạn đã là tiếng Việt thì dictionary này không cần thiết,
# nhưng nếu dùng yolov8n.pt thì nó sẽ dịch vài từ phổ biến
vi_translation = {
    'stop sign': 'Biển báo dừng',
    'person': 'Người',
    'car': 'Ô tô',
    'motorcycle': 'Xe máy',
    'bus': 'Xe buýt',
    'truck': 'Xe tải',
    'traffic light': 'Đèn giao thông'
}

@app.route('/')
def hello():
    return render_template('index.html')


@app.route('/image', methods=['POST'])
def predict():
    if request.method == 'POST':
        # 1. Nhận và lưu ảnh
        file = request.files['image_name']
        img_path = os.path.join(os.getcwd(), 'static', 'upload_image.jpg')
        file.save(img_path)

        # 2. Xử lý qua YOLOv8
        results = model.predict(source=img_path)
        
        # 3. Lưu ảnh kết quả (YOLOv8 tự vẽ bounding box)
        result = results[0]
        result_img_path = os.path.join(os.getcwd(), 'static', 'result.jpg')
        result.save(filename=result_img_path)
        
        # 4. Trích xuất tên class được nhận diện
        detected_names_set = set() # dùng set để loại bỏ danh sách trùng lặp
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = model.names[class_id]
            # Dịch sang Tiếng Việt nếu có trong từ điển
            if class_name in vi_translation:
                class_name = vi_translation[class_name]
            detected_names_set.add(class_name)
        
        detected_names = list(detected_names_set)

        # 5. Tạo file audio đọc bằng giọng nói
        if len(detected_names) > 0:
            text_to_speak = "Hệ thống phát hiện thấy: " + ", ".join(detected_names)
        else:
            text_to_speak = "Hệ thống không phát hiện thấy đối tượng nào"
            
        print("Đọc văn bản: ", text_to_speak)
        
        # Gọi gTTS sinh file mp3
        tts = gTTS(text=text_to_speak, lang='vi')
        audio_path = os.path.join(os.getcwd(), 'static', 'voice.mp3')
        # Ghi đè file voice cũ
        if os.path.exists(audio_path):
            os.remove(audio_path)
        tts.save(audio_path)

        return render_template('output.html', detected_names=detected_names)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)