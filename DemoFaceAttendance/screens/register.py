import os
from PyQt6.QtWidgets import *
from processor import process_frame
from screens.base import BaseScreen
import numpy as np
from database.user import save_user

class Register(BaseScreen):
    """Class xử lý đăng ký khuôn mặt"""
    
    def __init__(self, cap, fps, width, height, emb_path, max_captures=30):
        super().__init__(cap, fps, width, height, emb_path)
        self.current_name = ""
        self.is_capturing = False
        self.capture_count = 0
        self.max_captures = max_captures
        self.embeddings = []
        self.init_ui()
    
    def init_ui(self):
        """Khởi tạo giao diện đăng ký"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Camera label
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.width, self.height)
        self.camera_label.setStyleSheet("margin: 0px; padding: 0px;")
        layout.addWidget(self.camera_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        
        # Form đăng ký - Input bên trái, Button bên phải
        form_layout = QHBoxLayout()
        form_layout.setContentsMargins(10, 10, 10, 10)
        form_layout.setSpacing(10)
        
        # Input ở bên trái
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Nhập tên muốn đăng ký...")
        self.name_input.setMinimumHeight(40)
        self.name_input.setStyleSheet("font-size: 16px; padding: 5px;")
        self.name_input.returnPressed.connect(self.start_capture)
        form_layout.addWidget(self.name_input)
        
        # Nút ở bên phải
        self.start_btn = QPushButton("Bắt đầu đăng ký")
        self.start_btn.clicked.connect(self.start_capture)
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setMinimumWidth(150)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        form_layout.addWidget(self.start_btn)
        
        layout.addLayout(form_layout)
        
        self.setLayout(layout)
    
    def start_capture(self):
        """Bắt đầu chụp ảnh"""
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Lỗi", "Vui lòng nhập tên!")
            return
        
        self.current_name = name
        self.is_capturing = True
        self.capture_count = 0
        self.embeddings = []
        # Ẩn form, hiện progress bar
        self.name_input.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        
        # Bắt đầu timer
        self.start()
        
        print(f"Bắt đầu đăng ký cho: {name}")
    
    def update(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Hiển thị + lấy embedding
        embedding = process_frame(self.camera_label, frame)

        # Thu embedding nếu đang capture
        if self.is_capturing and embedding is not None:
            self.embeddings.append(embedding)

            self.capture_count += 1
            progress = int((self.capture_count / self.max_captures) * 100)
            self.progress_bar.setValue(progress)

            # Đủ số lượng embedding thì dừng
            if self.capture_count >= self.max_captures:
                self.finish_capture()

    def save_embedding(self):
        if not self.embeddings: return
        label = self.current_name
        # gộp embedding
        final_embedding = np.mean(self.embeddings, axis=0)
        final_embedding = final_embedding / np.linalg.norm(final_embedding)
        final_embedding = final_embedding.astype("float32")
        os.makedirs("embeddings", exist_ok=True)
        if os.path.exists(self.emb_path):
            data = np.load(self.emb_path, allow_pickle=True)
            embeddings = data["embeddings"]
            labels = data["labels"].tolist()
        else:
            embeddings = np.empty((0, final_embedding.shape[0]), dtype="float32")
            labels = []

        if label in labels:
            idx = labels.index(label)
            embeddings[idx] = final_embedding
        else:
            embeddings = np.vstack([embeddings, final_embedding])
            labels.append(label)

        np.savez(self.emb_path, embeddings=embeddings, labels=np.array(labels))
        self.embeddings = []

    def finish_capture(self):
        """Kết thúc quá trình chụp"""
        self.is_capturing = False
        # KHÔNG stop timer - để camera chạy tiếp
        self.save_embedding()
        
        QMessageBox.information(self, "Thành công", f"Đã hoàn tất đăng ký cho {self.current_name}!")
        
        # Reset form
        self.name_input.clear()
        self.name_input.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.progress_bar.hide()
        save_user(self.current_name)
        print(f"Hoàn tất đăng ký cho {self.current_name}!")
    
    def stop(self):
        """Override stop để xử lý thêm"""
        super().stop()
        self.is_capturing = False
    
    def start(self):
        """Override start để reset form nếu cần"""
        super().start()
        if not self.is_capturing:
            self.name_input.setEnabled(True)
            self.start_btn.setEnabled(True)
            self.progress_bar.hide()