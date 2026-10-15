from PyQt6.QtWidgets import QVBoxLayout, QLabel, QMessageBox, QSpacerItem, QSizePolicy
from PyQt6.QtCore import QTimer
from processor import process_frame
from screens.base import BaseScreen
import numpy as np
import os
import time
from collections import Counter
from database.attendance import save_attendance

class Attendance(BaseScreen):
    """Class xử lý chấm công"""
    
    def __init__(self, cap, fps, width, height, emb_path, ratio=0.8, threshold=0.5, time_consuming=3, cooldown_seconds=2):
        super().__init__(cap, fps, width, height, emb_path)
        self.init_ui()
        self.collecting = False
        self.collect_start_time = None
        self.label_buffer = []
        self.popup_shown = False
        self.ratio = ratio
        self.threshold = threshold
        self.time_consuming = time_consuming
        self.cooldown_until = 0
        self.cooldown_seconds = cooldown_seconds
        self.start() 
    
    def init_ui(self):
        """Khởi tạo giao diện chấm công"""
        layout = QVBoxLayout()
        
        # Camera label - sử dụng self.width và self.height
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(self.width, self.height)
        layout.addWidget(self.camera_label)
        self.setLayout(layout)

    def recognize_embedding(self, query_embedding, threshold):
        if not os.path.exists(self.emb_path): return None, 0.0

        data = np.load(self.emb_path, allow_pickle=True)
        db_embeddings = data["embeddings"]
        db_labels = data["labels"].tolist()

        if len(db_embeddings) == 0: return None, 0.0
        query = query_embedding / np.linalg.norm(query_embedding)
        sims = np.dot(db_embeddings, query)
        best_idx = np.argmax(sims)
        best_score = float(sims[best_idx])
        if best_score < threshold: return "unknown", best_score
        return db_labels[best_idx], best_score
    
    def check_majority_label(self):
        if len(self.label_buffer) == 0:
            return None

        counter = Counter(self.label_buffer)
        label, count = counter.most_common(1)[0]
        ratio = count / len(self.label_buffer)

        if ratio >= self.ratio and label != "unknown":
            return label
        return None

    def show_success_popup(self, label):
        save_attendance(label)
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(f"Đã chấm công cho {label}!")
        msg.setWindowTitle("Chấm công thành công")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        layout = msg.layout()
        spacer = QSpacerItem(300, 0,QSizePolicy.Policy.Minimum,QSizePolicy.Policy.Expanding)
        layout.addItem(spacer, layout.rowCount(), 0, 1, layout.columnCount())
        QTimer.singleShot(1500, msg.accept)
        msg.exec()

    def start(self):
        super().start()
        self.cooldown_until = time.time() + self.cooldown_seconds
        self.collecting = False
        self.collect_start_time = None
        self.label_buffer.clear()
        self.popup_shown = False

    def update(self):
        ret, frame = self.cap.read()
        if not ret: return
        embedding = process_frame(self.camera_label, frame)
        if time.time() < self.cooldown_until:
            return

        if embedding is not None:
            label, score = self.recognize_embedding(embedding, self.threshold)
            self.last_label = label
            self.last_score = score
        else:
            self.last_label = None
            self.last_score = 0.0
            return
        current_time = time.time()

        if not self.collecting and label not in (None, "unknown"):
            self.collecting = True
            self.collect_start_time = current_time
            self.label_buffer = []
            self.popup_shown = False

        if self.collecting:
            if current_time - self.collect_start_time <= self.time_consuming:
                self.label_buffer.append(label)
            else:
                final_label = self.check_majority_label()

                if final_label and not self.popup_shown:
                    self.show_success_popup(final_label)
                    self.popup_shown = True

                self.collecting = False
                self.collect_start_time = None
                self.label_buffer.clear()
