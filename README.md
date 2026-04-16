# 🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông Việt Nam bằng AI

Chào mừng đến với dự án **Nhận diện biển báo giao thông Việt Nam** của **Nhóm 7**. Dự án này ứng dụng công nghệ trí tuệ nhân tạo, cụ thể là mạng nơ-ron tích chập (YOLOv8), để phát hiện và nhận diện các loại biển báo giao thông trên đường phố Việt Nam. Đặc biệt, hệ thống còn tích hợp công nghệ Text-to-Speech (gTTS) để tự động phát âm nhắc nhở bằng giọng nói tiếng Việt các biển báo đã nhận diện được, hỗ trợ đắc lực cho người tham gia giao thông.

---

## 📑 Mục Lục
1. [Giới thiệu Dự Án](#-giới-thiệu-dự-án)
2. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
3. [Tính Năng Chính](#-tính-năng-chính)
4. [Môi Trường & Cài Đặt](#-môi-trường--cài-đặt)
5. [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
   - [5.1 Tiền xử lý dữ liệu & Data Augmentation](#51-tiền-xử-lý-dữ-liệu--data-augmentation)
   - [5.2 Huấn luyện mô hình (Train)](#52-huấn-luyện-mô-hình-train)
   - [5.3 Chạy Ứng dụng Web / Giao diện Phát Hiện](#53-chạy-ứng-dụng-web--giao-diện-phát-hiện)
6. [Danh Sách Lớp Nhận Diện](#-danh-sách-lớp-nhận-diện)
7. [Thành Viên](#-thành-viên)

---

## 🎯 Giới thiệu Dự Án

Dự án này là quy trình từ A-Z để thiết lập, huấn luyện và tối ưu một hệ thống Computer Vision. Khung làm việc đi từ bước chuẩn bị, tiền xử lý dữ liệu, huấn luyện mô hình YOLOv8, cho đến việc triển khai một ứng dụng web (Flask) cho end-user. Người dùng có thể upload một hình ảnh, AI sẽ dự đoán biển báo, đánh dấu vùng phát hiện (bounding box) và thông báo qua âm thanh tiếng Việt.

---

## 📂 Cấu Trúc Thư Mục

Dự án được phân rã thành các thư mục với chức năng chuyên biệt:

```text
AI-NhanDienBienBaoVN-Nhom7/
│
├── API/                            # (Thư mục chứa mã nguồn triển khai ứng dụng Web)
│   ├── static/                     # Chứa tài nguyên tĩnh (ảnh upload, ảnh kết quả, file âm thanh .mp3)
│   ├── templates/                  # Chứa giao diện HTML front-end (index.html, output.html)
│   ├── app.py                      # Flask Server API chạy ứng dụng
│   └── requirements.txt            # Chứa các thư viện và version cần thiết
│
├── dataset/                        # (Thư mục cấu hình và dữ liệu huấn luyện)
│   └── data.yaml                   # File cấu hình chứa link dataset và 56 classes biển báo VN
│
├── Source code/                    # (Thư mục chứa script phân tích và huấn luyện model)
│   ├── data_preprocessing_demo.py  # Script minh họa tiền xử lý dữ liệu và tạo báo cáo trực quan
│   ├── train_yolov8.py             # Script thiết lập huấn luyện YOLOv8
│   └── yolov8n.pt                  # Pre-trained weights YOLOv8 nano
│
└── README.md                       # File mô tả dự án
```

---

## 🚀 Tính Năng Chính

*   **Chẩn đoán Dữ Liệu (Data Preprocessing):** Tính toán Histogram màu sắc, Canny Edge Detection, cùng các kỹ thuật tăng cường dữ liệu (Data Augmentation) như xoay, lật, tăng độ sáng, làm mờ Gaussian để xử lý mọi trường hợp của camera thực tế.
*   **Huấn Luyện AI (Training YOLOv8):** Hỗ trợ code huấn luyện model từ bộ dữ liệu có 56 nhãn biển báo ở Việt Nam, sử dụng cấu hình tối ưu.
*   **Web App Giao diện Thân thiện:** Sử dụng framework Flask để chạy một ứng dụng cho phép người dùng up ảnh lên trực tiếp từ thiết bị.
*   **Hỗ trợ Giọng nói Tiếng Việt (Text-to-Speech):** Tự động phân tích các biển báo trong khung hình, nối lại thành chuỗi văn bản và đọc cảnh báo người dùng qua định dạng `.mp3`.

---

## 🛠 Môi Trường & Cài Đặt

**Yêu cầu hệ thống:**
- Python 3.8 trở lên
- Pip (Python Package Installer)

**Cách cài đặt:**

1. Clone lại dự án về máy:
   ```bash
   git clone https://github.com/maivanhao2110/NhanDienBienBaoVN.git
   cd AI-NhanDienBienBaoVN-Nhom7
   ```

2. Cài đặt các thư viện lõi (Dùng cho thư mục API):
   ```bash
   cd API
   pip install -r requirements.txt
   ```

3. (Tùy chọn) Cài đặt các thư viện cần dùng cho báo cáo phân tích thuật toán:
   ```bash
   pip install opencv-python numpy matplotlib
   ```

---

## 📖 Hướng Dẫn Sử Dụng

### 5.1 Tiền xử lý dữ liệu & Data Augmentation
Để quan sát cách hệ thống phân tích hình khối và xử lý nhiễu bằng OpenCV, bạn chạy:
```bash
cd "Source code"
python data_preprocessing_demo.py
```
**Kết quả:** Script sẽ sinh ra file hình ảnh báo cáo `BaoCao_Preprocessing_Cua_Toi.png` thể hiện trực quan quá trình xử lý, rất phục vụ cho tài liệu báo cáo của Nhóm! 

### 5.2 Huấn luyện mô hình (Train)
Nếu bạn đã chuẩn bị Dataset và bỏ vào thư mục `dataset/` (tuân thủ theo cấu trúc định nghĩa trong `data.yaml`), bạn có thể tiến hành huấn luyện AI:
```bash
cd "Source code"
python train_yolov8.py
```
**Kết quả:** File weight dùng để nhận diện chính xác nhất sẽ được lưu tại `runs/detect/traffic_sign_v8/weights/best.pt`. Copy file này thả vào thư mục `API` và đổi tên nó thành `best_v8.pt` để chuẩn bị cho Web App.

### 5.3 Chạy Ứng dụng Web / Giao diện Phát Hiện
Khởi động hệ thống Flask theo kịch bản Nhận Diện & Đọc Giọng Nói:
```bash
cd API
python app.py
```
**Sử dụng:** 
- Mở trình duyệt truy cập vào `http://localhost:5000` hoặc `http://127.0.0.1:5000`.
- Upload một hình ảnh có chứa biển báo.
- Xem kết quả khoanh vùng và nhấn vào Player âm thanh để nghe AI đọc tên các biển báo.

---

## 🏷 Danh Sách Lớp Nhận Diện

Mô hình hỗ trợ 56 loại biển báo tương thích chặt chẽ với Quy Chuẩn Giao Thông Đường Bộ Việt Nam, gồm các biển báo Nguy Hiểm (W), Rẽ/Hiệu Lệnh (R), Cấm (P) tiêu biểu:
`'DP-135', 'P-102', 'P-103a', 'P-103b', 'P-103c', 'P-104', 'P-106a', 'P-106b', 'P-107a', 'P-112', 'P-115', 'P-117', 'P-123a', 'P-123b', 'P-124a', 'P-124b', 'P-124c', 'P-127', 'P-128', 'P-130', 'P-131a', 'P-137', 'P-245a', 'R-301c', 'R-301d', 'R-301e', 'R-302a', 'R-302b', 'R-303', 'R-407a', 'R-409', 'R-425', 'R-434', 'S-509a', 'W-201a', 'W-201b', 'W-202a', 'W-202b', 'W-203b', 'W-203c', 'W-205a', 'W-205b', 'W-205d', 'W-207a', 'W-207b', 'W-207c', 'W-208', 'W-209', 'W-210', 'W-219', 'W-224', 'W-225', 'W-227', 'W-233', 'W-235', 'W-245a'`

---

*Cảm ơn đã quan tâm tới đề tài nghiên cứu AI của Nhóm 7!*
