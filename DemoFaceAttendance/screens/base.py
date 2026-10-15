from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer

class BaseScreen(QWidget):
    """Class cha chứa các thuộc tính chung cho tất cả màn hình"""
    
    def __init__(self, cap, fps, width, height, emb_path):
        super().__init__()
        self.cap = cap
        self.fps = fps
        self.width = width
        self.height = height
        self.emb_path = emb_path
        
        # Timer chung
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
    
    def update(self):
        """Override method này ở class con"""
        pass
    
    def start(self):
        """Khởi động timer"""
        self.timer.start(int(1000/self.fps))
    
    def stop(self):
        """Dừng timer"""
        self.timer.stop()