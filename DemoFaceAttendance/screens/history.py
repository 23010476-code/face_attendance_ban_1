from database.attendance import get_attendances
from database.user import get_users
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QTabWidget, QTableWidget, QTableWidgetItem, QLabel
from PyQt6.QtCore import Qt

class History(QWidget):
    """Màn hình xem lịch sử chấm công & danh sách user"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("LỊCH SỬ HỆ THỐNG")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size:20px; font-weight:bold;")
        layout.addWidget(title)

        self.tabs = QTabWidget()

        self.tabs.addTab(self.create_attendance_tab(), "Chấm công")
        self.tabs.addTab(self.create_users_tab(), "Tài khoản")

        layout.addWidget(self.tabs)
        self.setLayout(layout)

    def create_attendance_tab(self):
        table = QTableWidget()
        data = get_attendances()

        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["ID", "Tên", "Thời gian"])
        table.setRowCount(len(data))

        for row, (id_, label, timestamp) in enumerate(data):
            table.setItem(row, 0, QTableWidgetItem(str(id_)))
            table.setItem(row, 1, QTableWidgetItem(label))
            table.setItem(row, 2, QTableWidgetItem(timestamp))

        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        return table

    def create_users_tab(self):
        table = QTableWidget()
        data = get_users()

        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["ID", "Tên"])
        table.setRowCount(len(data))

        for row, (id_, label) in enumerate(data):
            table.setItem(row, 0, QTableWidgetItem(str(id_)))
            table.setItem(row, 1, QTableWidgetItem(label))

        table.resizeColumnsToContents()
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        return table

    def start(self):
        """Khi màn hình được show lại → reload dữ liệu"""
        self.tabs.clear()
        self.tabs.addTab(self.create_attendance_tab(), "Chấm công")
        self.tabs.addTab(self.create_users_tab(), "Tài khoản")

    def stop(self):
        return