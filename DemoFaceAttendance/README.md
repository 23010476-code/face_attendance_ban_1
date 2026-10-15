cd 'E:\nhap_mon_ai\face_attendance (1)\face_attendance\face-attendance-master'
& '.\myenv\Scripts\python.exe' -m uvicorn main:app --host 127.0.0.1 --port 8000

### 0. Tải dự án về:
- Đảm bảo máy có sẵn Visual Studio Code và Git
- Gõ ở Terminal: `git clone https://github.com/HieuCaoTlu/DemoFaceAttendance.git`
- Đi vào trong thư mục `cd .\DemoFaceAttendance`

### 1. Cài môi trường trên Windows
- Cài đặt uv - trình quản lý thư viện Python hiện đại:
+ Mở Powershell lên, gõ: `irm https://astral.sh/uv/install.ps1 | iex`
+ Bạn sẽ nhìn thấy dòng "To add ... to your PATH", copy và chạy đoạn code thứ 2
+ VD: `$env:Path = "C:\Users\yscfs\.local\bin;$env:Path"`
+ Kiểm tra xem uv cài đặt thành công chưa: `uv --version`

### 2. Khởi tạo dự án
+ Cài python 3.10: `uv python install 3.10`
+ Khởi tạo môi trường ảo: `uv venv --python 3.10`
+ Kích hoạt môi trường ảo: `.venv\Scripts\activate` (kích hoạt xong sẽ thấy có dấu mở ngoặc tên dự án ở con trỏ Terminal)
+ Những lần mở dự án sau cũng sẽ xài lệnh trên để kích hoạt môi trường ảo
+ Cài đặt thư viện: `uv sync`

### 3. Chạy ứng dụng lần sau:
+ `.venv\Scripts\activate ` (nếu đã kích hoạt thì không cần chạy)
+ `python app.py` (lần đầu khởi động thì sẽ mất thời gian 1 chút, những lần sau sẽ nhanh hơn)

### 4. Công nghệ sử dụng
- PyQt: Vẽ lên phần mềm desktop Python
- OpenCV: Mở camera và xử lý ảnh
- Mediapipe FaceDetection BlazeFace: mô hình nhận diện ảnh
- MobileFaceNet: mô hình trích xuất vector, sử dụng hàm mất mát ArcFace, được tiền huấn luyện ở database MS1MV2

### 5. Cấu trúc dự án:
#### Database: chứa các đối tượng cần lưu trữ
+ Attendance - Lịch sử chấm công: ID, Label, Timestamp
+ User - Người dùng: ID, Label

#### Screens: danh sách các màn hình của dự án
+ Base: khung dựng màn hình 
+ Attendance: màn hình chấm công (sau này sẽ thiết lập quy tắc chấm công tại đây)
+ Register: màn hình đăng ký khuôn mặt
+ History: xem các bảng trong database
+ Reset: xóa mọi dữ liệu embeddings

#### Models: danh sách mô hình sử dụng
+ detector.tflite: mô hình nhận dạng mặt của mediapipe (Mediapipe FaceDetection - BlazeFace(short-range))
+ inference: mô hình trích xuất vector từ ảnh mặt (ArcFace - MFN (paddle) - MS1MV2: [Link](https://github.com/deepinsight/insightface/tree/master/model_zoo))

#### File khác
+ pyproject.toml: nơi định nghĩa dự án và các thư viện cần cài đặt
+ database.db: cơ sở dữ liệu của dự án

### 6. Giải thích AI Pipeline
- Với bất cứ tác vụ chấm công hay đăng ký khuôn mặt, chương trình sử dụng 1 file tên là processor.py
    - Ảnh lấy được từ webcam liên tục được truyền vào hàm process_frame
    - Ảnh được lật cho đúng hướng, giảm nhiễu
    - Ảnh được đưa vào Detector Mediapipe để phát hiện mặt
    - Detector trả về tọa độ bbox để cắt được mặt từ ảnh
    - Mặt đã được cắt từ ảnh tiếp tục được đi trích xuất vector, trở thành Embedding
    - Ảnh được xử lý lại kênh màu trước khi hiển thị trên giao diện
    - Embedding sẽ được trả về nơi gọi hàm: Attendance hoặc Register.

- Cơ chế chấm công:
    - Mỗi khi nhận được embedding, Attendance sử dụng thuật toán **Cosine Similarity**. Thuật toán này so sánh embedding hiện tại với mọi embedding có trong file lưu trữ, rồi lấy embedding giống nhất. Độ tương đồng giữa 2 embedding sau đó được kiểm tra xem có vượt qua ngưỡng chấp nhận không? Nếu có thì lấy ra nhãn của embedding giống đó, hiển thị lên giao diện.
    - Khi bắt đầu dự đoán tên của người trước màn hình, Attendance sẽ thiết lập 1 lần kiểm tra mặt
    - Lần kiểm tra này sẽ kéo dài trong 3 giây (có thể tăng/giảm tại app.py)
    - Trong 3 giây này, nếu 80% kết quả dự đoán đều trùng nhãn, lần kiểm tra này coi như thành công
    - Khi thành công, lưu lại thời điểm chấm công vừa rồi vào cơ sở dữ liệu
    - **[Lưu ý: phần chấm công này chưa có cơ chế so sánh thời gian ca làm, bạn phải tự thiết lập]**

- Cơ chế đăng ký mặt:
    - Người dùng ban đầu điền tên mình vào, rồi click nút đăng ký
    - Register bắt đầu chụp liên tiếp để lấy ra 1 lượng embedding nhất định (có thể cài đặt số lượng tại app.py)
    - Khi lấy đủ, nó tính **trung bình toàn bộ embedding**, tạo ra **1 embedding duy nhất đại diện cho người dùng đang đăng ký**.
    - Sau đó lưu chúng vào trong bộ nhớ 1 bộ gồm (embedding trung bình, label), label là tên người dùng điền trên giao diện.
    - Trong trường hợp trùng label, nó sẽ ghi đè embedding cũ


