import cv2
from PyQt6.QtGui import QImage, QPixmap
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import paddle.inference as paddle_infer

base_options = python.BaseOptions(model_asset_path='./models/detector.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
config = paddle_infer.Config("./models/inference.pdmodel", "./models/inference.pdiparams")
config.disable_gpu()
config.disable_mkldnn()
predictor = paddle_infer.create_predictor(config)

def get_embedding(frame_rgb, bbox):
    x, y, bw, bh = bbox
    h, w, _ = frame_rgb.shape
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + bw)
    y2 = min(h, y + bh)
    face = frame_rgb[y1:y2, x1:x2]
    if face.size == 0: return None
    face = cv2.resize(face, (112, 112))
    face = face.astype("float32") / 255.0
    face = (face - 0.5) / 0.5            # [-1, 1]
    face = np.transpose(face, (2, 0, 1)) # HWC → CHW
    face = np.expand_dims(face, axis=0)  # [1, 3, 112, 112]
    input_name = predictor.get_input_names()[0]
    input_handle = predictor.get_input_handle(input_name)
    input_handle.reshape(face.shape)
    input_handle.copy_from_cpu(face)
    predictor.run()
    output_name = predictor.get_output_names()[0]
    output_handle = predictor.get_output_handle(output_name)
    embedding = output_handle.copy_to_cpu()[0]
    return embedding

def process_frame(display, frame):
    '''
    Xử lý ảnh trước hiển thị:
    - Lật lại ảnh để đúng hướng
    - Giảm nhiễu nhẹ
    - Đổi kênh màu để hiển thị
    - Truyền ảnh vào giao diện hiển thị
    '''
    frame = cv2.flip(frame, 1)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_display = frame_rgb.copy()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)
    bbox = None
    embedding = None
    if result.detections:
        detection = result.detections[0]
        b = detection.bounding_box
        x, y, bw, bh = b.origin_x, b.origin_y, b.width, b.height
        bbox = (x, y, bw, bh)
        cv2.rectangle(frame_display, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        embedding = get_embedding(frame_rgb, bbox)
        parent = display.parent()
        label = getattr(parent, "last_label", None)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            text = str(label)
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.putText(frame_display, text, (x + bw // 2 - tw // 2, y + bh + th + 10), font, scale, (255, 0, 0), thickness, cv2.LINE_AA)
            
    img = QImage(frame_display.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    display.setPixmap(QPixmap.fromImage(img))
    return embedding