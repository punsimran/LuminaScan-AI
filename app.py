import sys
import torch
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor
from PyQt5.QtCore import Qt

class MinimalPneumoniaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LuminaScan AI - Diagnostic Suite")
        self.resize(1000, 750)
        self.setStyleSheet("background-color: #FAFAFA;") 

        # Load Model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
            self.model.conf = 0.25 
        except Exception as e:
            print(f"Model Error: {e}")
            self.model = None

        self.initUI()
        self.image_path = None

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- SIDEBAR ---
        sidebar = QFrame()
        sidebar.setFixedWidth(220)
        sidebar.setStyleSheet("background-color: #FFFFFF; border-right: 1px solid #EEEEEE;")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(25, 40, 25, 40)

        logo = QLabel("PneumoniaAI")
        logo.setStyleSheet("font-size: 20px; font-weight: bold; color: #1A1A1A; border: none;")
        sidebar_layout.addWidget(logo)

        tagline = QLabel("Diagnostic Tool")
        tagline.setStyleSheet("font-size: 11px; color: #999999; margin-bottom: 50px; border: none;")
        sidebar_layout.addWidget(tagline)

        self.btn_upload = self.create_pill_button("Upload Scan", "#F0F0F0", "#333333")
        self.btn_upload.clicked.connect(self.open_file) # Opens computer file explorer
        sidebar_layout.addWidget(self.btn_upload)

        self.btn_detect = self.create_pill_button("Run Analysis", "#E8F5E9", "#2E7D32")
        self.btn_detect.setEnabled(False)
        self.btn_detect.clicked.connect(self.run_detection)
        sidebar_layout.addWidget(self.btn_detect)

        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)

        # --- MAIN VIEW ---
        view_area = QWidget()
        view_layout = QVBoxLayout(view_area)
        view_layout.setContentsMargins(50, 40, 50, 40)

        self.status_line = QLabel("Awaiting Scan")
        self.status_line.setStyleSheet("font-size: 13px; color: #AAAAAA; font-weight: 500;")
        view_layout.addWidget(self.status_line)

        self.image_container = QLabel("Select an X-ray from your computer")
        self.image_container.setAlignment(Qt.AlignCenter)
        self.image_container.setStyleSheet("background-color: #FFFFFF; border: 1px solid #EEEEEE; border-radius: 20px; color: #CCCCCC;")
        
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(15)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 15))
        self.image_container.setGraphicsEffect(shadow)
        view_layout.addWidget(self.image_container, 1)

        # --- RESULT BADGE ---
        self.result_badge = QFrame()
        self.result_badge.setFixedHeight(70)
        self.result_badge.setFixedWidth(450)
        self.result_badge.setStyleSheet("background-color: #FFFFFF; border-radius: 35px; border: 1px solid #EEEEEE;")
        badge_layout = QHBoxLayout(self.result_badge)

        self.dot = QLabel("●")
        self.dot.setStyleSheet("font-size: 20px; color: #DDDDDD; border: none;")
        badge_layout.addWidget(self.dot)

        self.res_label = QLabel("Diagnosis Pending")
        self.res_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #666666; border: none;")
        badge_layout.addWidget(self.res_label, 1)

        self.conf_label = QLabel("--%")
        self.conf_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #999999; border: none;")
        badge_layout.addWidget(self.conf_label)

        badge_container = QHBoxLayout()
        badge_container.addStretch()
        badge_container.addWidget(self.result_badge)
        badge_container.addStretch()
        
        view_layout.addSpacing(30)
        view_layout.addLayout(badge_container)
        main_layout.addWidget(view_area)

    def create_pill_button(self, text, bg, fg):
        btn = QPushButton(text)
        btn.setFixedHeight(45)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(f"QPushButton {{ background-color: {bg}; color: {fg}; border-radius: 12px; font-size: 13px; font-weight: 600; border: none; margin-bottom: 10px; }} QPushButton:hover {{ background-color: #E0E0E0; }}")
        return btn

    def open_file(self):
        # This opens your computer's local file explorer
        file_path, _ = QFileDialog.getOpenFileName(self, "Select X-ray from Computer", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_container.setPixmap(pixmap.scaled(self.image_container.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.btn_detect.setEnabled(True)
            self.status_line.setText("Scan Loaded Successfully")

    def run_detection(self):
        if not self.model: return
        
        results = self.model(self.image_path)
        rendered_img = results.render()[0] 
        q_img = QImage(rendered_img.data, rendered_img.shape[1], rendered_img.shape[0], 3 * rendered_img.shape[1], QImage.Format_RGB888)
        self.image_container.setPixmap(QPixmap.fromImage(q_img).scaled(self.image_container.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        detections = results.xyxy[0] # List of detections
        
        if len(detections) > 0:
            # Get the top detection (highest confidence)
            top_detection = detections[0]
            conf_score = top_detection[4].item() * 100
            class_idx = int(top_detection[5].item())
            class_name = self.model.names[class_idx] # Get label name (e.g. "Normal" or "Pneumonia")

            if "pneumonia" in class_name.lower():
                # PNEUMONIA UI (Red)
                self.dot.setStyleSheet("color: #FF5252; font-size: 20px; border: none;")
                self.res_label.setText("Pneumonia Detected")
                self.res_label.setStyleSheet("color: #D32F2F; font-size: 14px; font-weight: 700; border: none;")
                self.conf_label.setText(f"{conf_score:.1f}%")
                self.conf_label.setStyleSheet("color: #D32F2F; font-size: 14px; font-weight: 800; border: none;")
                self.result_badge.setStyleSheet("background-color: #FFEBEE; border-radius: 35px; border: 1px solid #FFCDD2;")
            else:
                # NORMAL UI (Green)
                self.dot.setStyleSheet("color: #4CAF50; font-size: 20px; border: none;")
                self.res_label.setText("Normal Presentation")
                self.res_label.setStyleSheet("color: #2E7D32; font-size: 14px; font-weight: 700; border: none;")
                self.conf_label.setText(f"{conf_score:.1f}%")
                self.conf_label.setStyleSheet("color: #2E7D32; font-size: 14px; font-weight: 800; border: none;")
                self.result_badge.setStyleSheet("background-color: #E8F5E9; border-radius: 35px; border: 1px solid #C8E6C9;")
        else:
            self.res_label.setText("Inconclusive Scan")
            self.status_line.setText("No features detected by the model.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MinimalPneumoniaApp()
    window.show()
    sys.exit(app.exec_())