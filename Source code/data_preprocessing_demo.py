import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def main():
    print("Đang khởi tạo công cụ phân tích và xử lý dữ liệu...")
    
    # 1. Cài đặt đường dẫn tới thư mục dataset
    # Trỏ tới thư mục chứa tập train của dataset đã tải về
    dataset_path = '../dataset/train/images'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Lỗi: Không tìm thấy thư mục {dataset_path}. Vui lòng kiểm tra lại vị trí đặt code.")
        return

    # Lọc lấy danh sách các file ảnh
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        print("❌ Lỗi: Không tìm thấy ảnh nào trong thư mục dataset!")
        return

    # Chọn ngẫu nhiên 1 ảnh biển báo trong cục 6700 ảnh để làm mẫu phân tích
    sample_image = random.choice(image_files)
    img_path = os.path.join(dataset_path, sample_image)
    print(f"👉 Đang phân tích ảnh mẫu: {sample_image}")

    # ==========================================
    # PHẦN 1: TẢI ẢNH & HIỂN THỊ CƠ BẢN
    # ==========================================
    # Đọc ảnh bằng OpenCV (Mặc định OpenCV đọc theo hệ màu BGR)
    image_bgr = cv2.imread(img_path)
    # Chuyển đổi màu từ BGR sang RGB để hiển thị đúng thực tế trên Matplotlib
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Thiết lập khung vẽ biểu đồ báo cáo
    plt.figure(figsize=(16, 10))
    plt.suptitle(f"Báo cáo Data Preprocessing & Feature Engineering\nFile mẫu: {sample_image}", fontsize=16, fontweight='bold')

    # Ô số 1: Ảnh gốc
    plt.subplot(2, 4, 1)
    plt.imshow(image_rgb)
    plt.title("1. Ảnh gốc (Original RGB)")
    plt.axis('off')

    # ==========================================
    # PHẦN 2: FEATURE ENGINEERING (Tính toán đặc trưng)
    # ==========================================
    
    # Minh chứng 2.1: Phân tích Màu sắc (Color Histogram)
    # Rất quan trọng vì biển báo giao thông phụ thuộc nhiều vào màu Đỏ/Xanh lam/Vàng
    plt.subplot(2, 4, 2)
    colors = ('r', 'g', 'b')
    for i, col in enumerate(colors):
        # Tính toán phân phối pixel cho từng kênh màu
        hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title("2. Đặc trưng: Color Histogram")
    plt.xlabel("Giá trị Pixel (0-255)")
    plt.ylabel("Tần suất xuất hiện")

    # Minh chứng 2.2: Trích xuất đường biên/viền hình khối (Canny Edge Detection)
    # Giúp chứng minh AI nhận ra được hình tròn/tam giác của biển báo
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred_gray = cv2.GaussianBlur(gray_image, (5, 5), 0) # Làm mờ giảm nhiễu trước khi lấy biên
    edges = cv2.Canny(blurred_gray, threshold1=100, threshold2=200)

    plt.subplot(2, 4, 3)
    plt.imshow(edges, cmap='gray')
    plt.title("3. Đặc trưng: Edge Detection (Canny)")
    plt.axis('off')

    # ==========================================
    # PHẦN 3: DATA AUGMENTATION (Tăng cường dữ liệu)
    # Mở rộng dataset thực tế (mô phỏng kỹ thuật YOLO dataloader)
    # ==========================================

    # Minh chứng 3.1: Quay ảnh (Rotation) - Mô phỏng góc cam bị nghiêng
    rows, cols = image_rgb.shape[:2]
    # Xoay ngẫu nhiên nghiêng 15 độ
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1) 
    rotated_image = cv2.warpAffine(image_rgb, rotation_matrix, (cols, rows))

    plt.subplot(2, 4, 4)
    plt.imshow(rotated_image)
    plt.title("4. Augment: Xoay (Rotate 15°)")
    plt.axis('off')

    # Minh chứng 3.2: Lật ảnh ngang (Horizontal Flip)
    flipped_image = cv2.flip(image_rgb, 1)

    plt.subplot(2, 4, 5)
    plt.imshow(flipped_image)
    plt.title("5. Augment: Lật ảnh ngang (Flip)")
    plt.axis('off')

    # Minh chứng 3.3: Làm nhòe/mờ (Gaussian Blur) - Mô phỏng sương mù/cam out nét
    blurred_image = cv2.GaussianBlur(image_rgb, (15, 15), 0)

    plt.subplot(2, 4, 6)
    plt.imshow(blurred_image)
    plt.title("6. Augment: Làm nhòe (Gaussian Blur)")
    plt.axis('off')

    # Minh chứng 3.4: Thay đổi độ sáng (Brightness & Contrast) - Mô phỏng thiếu sáng/chói nắng
    bright_image = cv2.convertScaleAbs(image_rgb, alpha=1.2, beta=40)

    plt.subplot(2, 4, 7)
    plt.imshow(bright_image)
    plt.title("7. Augment: Tăng độ sáng chói")
    plt.axis('off')

    # ==========================================
    # PHẦN 4: FINAL PREPROCESSING (Dành cho AI)
    # ==========================================
    # Resize về 640x640 (chuẩn của YOLOv8) và Normalize [0-1]
    resized_image = cv2.resize(image_rgb, (640, 640))
    normalized_image = resized_image / 255.0 # Chuẩn hóa pixel về 0.0 - 1.0
    
    # Đoạn này để hiển thị trực quan (matplotlib hiểu dải 0-1)
    plt.subplot(2, 4, 8)
    plt.imshow(normalized_image)
    plt.title("8. Preprocess: Resize 640x640 & Normalize")
    plt.axis('off')

    # ==========================================
    # LƯU VÀ HIỂN THỊ BÁO CÁO KẾT QUẢ
    # ==========================================
    plt.tight_layout()
    # Tự động canh lề title
    plt.subplots_adjust(top=0.9) 
    
    output_filename = "BaoCao_Preprocessing_Cua_Toi.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ HOÀN TẤT! Đã xuất ảnh báo cáo phân tích ra file gốc: {output_filename}")
    print("Chụp file ảnh này và cho vào slide/report để báo cáo phân việc của bạn!")
    
    # Hiển thị lên màn hình máy tính (nếu chạy local)
    plt.show()

if __name__ == "__main__":
    main()
