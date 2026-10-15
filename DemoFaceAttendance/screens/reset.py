import os
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QMessageBox

class Reset(QWidget):
    """Màn hình xóa toàn bộ dữ liệu embedding (.npz)"""

    def __init__(self, emb_path):
        super().__init__()
        self.emb_path = emb_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addStretch()

        title = QLabel("Bạn chắc chắn muốn xóa dữ liệu chứ?")
        title.setStyleSheet("font-size:18px; font-weight:bold;")

        btn_clear = QPushButton("Xóa")
        btn_clear.setMinimumHeight(50)
        btn_clear.clicked.connect(self.clear_file)

        layout.addWidget(title)
        layout.addSpacing(20)
        layout.addWidget(btn_clear)
        layout.addStretch()

        self.setLayout(layout)

    def start(self):
        return

    def stop(self):
        return

    def clear_file(self):
        if not os.path.exists(self.emb_path):
            QMessageBox.information(self, "Thông báo", "Không có dữ liệu để xóa.")
            return

        try:
            os.remove(self.emb_path)
            QMessageBox.information(self, "Thành công", "Đã xóa dữ liệu.")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))