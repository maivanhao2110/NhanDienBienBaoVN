# 🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông Việt Nam bằng AI

Chào mừng đến với dự án **Nhận diện biển báo giao thông Việt Nam** của **Nhóm 7**. Dự án này ứng dụng công nghệ trí tuệ nhân tạo, cụ thể là mạng nơ-ron tích chập (YOLOv8), để phát hiện và nhận diện các loại biển báo giao thông trên đường phố Việt Nam. 

Hệ thống đã được nâng cấp với giao diện Single-Page Application cực kì mượt mà cho phép xử lý 3 chế độ: **Ảnh / Video / Real-time WebCam**. Đặc biệt, bộ máy có chức năng lọc rác, tích hợp Web Speech API đọc tên biển báo thời gian thực và mô-đun AI Đọc Chữ (EasyOCR) tự động bắt thông số tốc độ của biển báo `P-127` một cách chính xác nhất!

---

## 📑 Mục Lục
1. [Giới thiệu Dự Án](#-giới-thiệu-dự-án)
2. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)
3. [Tính Năng Chính](#-tính-năng-chính)
4. [Môi Trường & Cài Đặt](#-môi-trường--cài-đặt)
5. [Hướng Dẫn Sử Dụng](#-hướng-dẫn-sử-dụng)
6. [Danh Sách Lớp Nhận Diện](#-danh-sách-lớp-nhận-diện)

---

## 🎯 Giới thiệu Dự Án

Dự án này là quy trình từ A-Z để thiết lập, huấn luyện và tối ưu một hệ thống Computer Vision. Khung làm việc đi từ bước chuẩn bị, tiền xử lý dữ liệu, huấn luyện mô hình YOLOv8, cho đến việc triển khai một ứng dụng web (Flask) cho end-user. 
Người dùng không cần tải file hay reload trang, hệ thống sẽ tự động bắt luồng Frames từ video/camera của bạn, phân tích bằng YOLOv8 tại Backend, đẩy sang OCR (nếu là biển tốc độ), vẽ khung (Bounding box) bằng Canvas Overlay và đọc ra tiếng Việt!

---

## 📂 Cấu Trúc Thư Mục

Dự án được phân rã thành các thư mục với chức năng chuyên biệt:

```text
AI-NhanDienBienBaoVN-Nhom7/
│
├── API/                            # Mã nguồn ứng dụng Web
│   ├── static/                     # CSS, Script xử lý Camera (main.js)
│   ├── templates/                  # Giao diện HTML (index.html SPA)
│   ├── app.py                      # Flask Server (Nhận diện YOLO + OCR)
│   └── requirements.txt            # Thư viện yêu cầu cài đặt
│
├── dataset/                        # Dữ liệu huấn luyện
│   └── data.yaml                   # File cấu hình chứa 56 classes biển báo VN
│
├── Source code/                    # Script huấn luyện
│   ├── train_yolov8.py             # Script đào tạo model YOLOv8
│   └── yolov8n.pt                  # Pre-trained core weights dùng để train
│
├── setup_yolov8.bat                # Script tạo môi trường (virtual env) tự động
├── train.bat                       # Lệnh tắt chạy huấn luyện
└── README.md                       # File mô tả dự án (bạn đang đọc)
```

---

## 🚀 Tính Năng Chính

*   **Bộ Dữ Liệu Việt Nam:** Tích hợp bộ cấu hình gán nhãn gồm 56 mã biển báo đặc thù tuân thủ sát sao theo luật giao thông Việt Nam.
*   **Huấn Luyện AI (Training YOLOv8):** Hỗ trợ code huấn luyện model Object Detection (YOLOv8) gọn gàng, tự động xuất file model hoàn thiện cuối cùng.
*   **Web App 3-in-1 (SPA):** Giao diện một trang cực mượt chuyển đổi 3 chế độ: Upload Ảnh tĩnh, phân tích Video offline, và Track thời gian thực (Real-time). Tọa độ Box được đẩy qua fetch API nhanh chóng mà không cần reload trang.
*   **Bộ Não Kép (YOLO Cắm OCR):** Không chỉ nhận diện "Hạn chế tốc độ", hệ thống kết hợp thư viện `EasyOCR` chạy song song để tự động cắt nhỏ biển báo và móc nối con số bên trong trả về cho người dùng (Ví dụ: "Hạn chế tốc độ tối đa 50 km/h").
*   **Cảnh báo Voice Thông Minh:** Tích hợp Web Speech API phát âm Tiếng Việt tự nhiên cực lẹ. Có tính năng Debounce (kiểm soát thư giãn lặp giọng) giúp máy tự động nín và không đọc lặp lại cái biển đó nếu đã đọc cách đây 10 giây.

---

## 🛠 Môi Trường & Cài Đặt

**Yêu cầu hệ thống:** Python 3.8+ và Pip.

**Cách cài đặt:**

1. Clone dự án về máy:
   ```bash
   git clone https://github.com/maivanhao2110/NhanDienBienBaoVN.git
   cd AI-NhanDienBienBaoVN-Nhom7
   ```

2. Cài đặt các thư viện lõi cho Backend Web:
   ```bash
   cd API
   pip install -r requirements.txt
   ```
   *Lưu ý: `EasyOCR` sẽ tự cài đặt cùng và mất thêm chút thời gian để tải model trong lần khởi chạy File API đầu tiên.*

---

## 📖 Hướng Dẫn Sử Dụng

### 5.1 Training Mô hình YOLOv8
**Huấn luyện mô hình:** Nếu có chuẩn bị sẵn file dữ liệu bỏ vào thư mục `dataset/` (tuân thủ theo `data.yaml`), bạn có thể tiến hành huấn luyện AI dễ dàng thông qua script:
```bash
# Hoặc đơn giản là ấn chạy file "train.bat" ngoài màn hình
cd "Source code"
python train_yolov8.py
```
*(Kết quả file nhận diện cuối cùng là weight xuất sắc nhất sẽ nằm ở đường dẫn `runs/detect/.../best.pt`).*

Trưởng nhóm vui lòng chép/copy file `best.pt` này sau khi máy luyện xong vào cùng thư mục `API`, rồi ấn đổi tên thành file `best_v8.pt` để sẵn sàng làm "não bộ nhận diện chính" cho Website!

### 5.2 Chạy Ứng dụng Bắt biển báo
Từ thư mục API, khởi chạy máy chủ:
```bash
python app.py
```
**Sử dụng:** 
- Mở trình duyệt truy cập vào `http://localhost:5000`. Hệ thống có chế độ Dark / Glassmorphism UI.
- Bạn có thể chuyển bấm qua lại giữa 3 mục: Tải ảnh, Tải Video hoặc Bấm nút Bật WebCam. Đưa biển báo vào và tận hưởng sức mạnh phân tích của YOLO.

---

## 🏷 Danh Sách Lớp Nhận Diện (56 Classes)

Hệ thống được giới hạn ngặt nghèo trong phạm vi 56 mã biển báo đặc trưng của Luật giao thông đường bộ Việt Nam. Bất kỳ sự vật / nhiễu nào không thuộc danh sách này đều bị bộ lọc máy chủ bóc rụng để hạn chế đọc sai:

| Mã Biển | Phân Loại & Dịch Nghĩa | Mã Biển | Phân Loại & Dịch Nghĩa |
| :--- | :--- | :--- | :--- |
| **DP-135** | Hết tất cả các lệnh cấm | **R-303** | Nơi giao nhau vòng xuyến |
| **P-102** | Cấm đi ngược chiều | **R-407A** | Đường một chiều |
| **P-103A** | Cấm ô tô | **R-409** | Chỗ quay xe |
| **P-103B** | Cấm ô tô rẽ phải | **R-425** | Bệnh viện |
| **P-103C** | Cấm ô tô rẽ trái | **R-434** | Bến xe buýt |
| **P-104** | Cấm mô tô | **S-509A** | Chiều cao an toàn |
| **P-106A** | Cấm xe tải | **W-201A** | Chỗ ngoặt vòng bên trái |
| **P-106B** | Cấm xe tải lớn | **W-201B** | Chỗ ngoặt vòng bên phải |
| **P-107A** | Cấm ô tô khách và ô tô tải | **W-202A** | Nhiều chỗ ngoặt nguy hiểm |
| **P-112** | Cấm người đi bộ | **W-202B** | Nhiều chỗ ngoặt nguy hiểm |
| **P-115** | Hạn chế tải trọng | **W-203B** | Đường hẹp bên trái |
| **P-117** | Hạn chế chiều cao | **W-203C** | Đường hẹp bên phải |
| **P-123A** | Cấm rẽ trái | **W-205A** | Đường giao nhau cùng cấp |
| **P-123B** | Cấm rẽ phải | **W-205B** | Đường giao nhau cùng cấp |
| **P-124A** | Cấm quay đầu xe | **W-205D** | Đường giao nhau cùng cấp |
| **P-124B** | Cấm ô tô quay đầu xe | **W-207A** | Giao nhau với đường không ưu tiên |
| **P-124C** | Cấm rẽ trái và quay đầu xe | **W-207B** | Giao nhau với đường không ưu tiên |
| **P-127** | Hạn chế tốc độ tối đa | **W-207C** | Giao nhau với đường không ưu tiên |
| **P-128** | Cấm bấm còi | **W-208** | Giao nhau với đường ưu tiên |
| **P-130** | Cấm dừng và đỗ xe | **W-209** | Giao nhau có tín hiệu đèn |
| **P-131A** | Cấm đỗ xe | **W-210** | Giao nhau đường sắt có rào |
| **P-137** | Cấm rẽ trái và rẽ phải | **W-219** | Dốc xuống nguy hiểm |
| **P-245A** | Đi chậm | **W-224** | Đường người đi bộ cắt ngang |
| **R-301C** | Chỉ được rẽ trái | **W-225** | Trẻ em |
| **R-301D** | Chỉ được rẽ phải | **W-227** | Công trường |
| **R-301E** | Chỉ được rẽ trái | **W-233** | Nguy hiểm khác |
| **R-302A** | Vòng chướng ngại vật sang phải | **W-235** | Đường đôi |
| **R-302B** | Vòng chướng ngại vật sang trái | **W-245A** | Đi chậm |

---
*Báo cáo nghiên cứu Trí Tuệ Nhân Tạo được thực hiện bởi Nhóm 7.*
