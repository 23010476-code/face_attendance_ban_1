import os
import cv2
import joblib
import numpy as np
import random
import mediapipe as mp
import onnxruntime as ort

from tools import augmentate
from sklearn.svm import LinearSVC
from database import session, Embedding
from sklearn.calibration import CalibratedClassifierCV


# =========================================================
# MODEL PATH
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(BASE_DIR, "models")

rec_path = os.path.join(model_dir, "quantized_model.onnx")
cls_path = os.path.join(model_dir, "face_classifier.pkl")

os.makedirs(model_dir, exist_ok=True)


# =========================================================
# MEDIAPIPE
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    print("Đang tải model...")

    if not os.path.exists(rec_path):
        raise FileNotFoundError(
            f"Không tìm thấy model FaceNet: {rec_path}"
        )

    rec_model = ort.InferenceSession(rec_path)

    print("✅ Model ONNX sẵn sàng")

    if os.path.exists(cls_path):

        try:
            cls_model = joblib.load(cls_path)

            print("✅ Model SVC sẵn sàng")
            print("Classes:", cls_model.classes_)

        except Exception as e:

            print("❌ Không load được model SVC:", e)

            cls_model = None

    else:

        print("⚠️ Chưa có model SVC")

        cls_model = None

    return rec_model, cls_model


rec_model, cls_model = load_model()

cls_ready = cls_model is not None


# =========================================================
# DETECT FACE
# =========================================================

def detect_face(image):

    global mp_face_mesh, face_mesh

    if image is None or image.size == 0:
        return None, None

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(image_rgb)

    if (
        results.multi_face_landmarks
        and len(results.multi_face_landmarks) > 0
    ):

        face_landmarks = (
            results.multi_face_landmarks[0].landmark
        )

        img_h, img_w, _ = image.shape

        x_coords = [
            int(landmark.x * img_w)
            for landmark in face_landmarks
        ]

        y_coords = [
            int(landmark.y * img_h)
            for landmark in face_landmarks
        ]

        x1 = max(0, min(x_coords))
        x2 = min(img_w, max(x_coords))

        y1 = max(0, min(y_coords))
        y2 = min(img_h, max(y_coords))

        bbox_width = x2 - x1
        bbox_height = y2 - y1

        # Khuôn mặt quá nhỏ
        if min(bbox_width, bbox_height) < 70:
            return None, None

        # Bounding box không hợp lệ
        if x2 <= x1 or y2 <= y1:
            return None, None

        face = image[y1:y2, x1:x2]

        if face is None or face.size == 0:
            return None, None

        bbox = np.array(
            [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2)
            ],
            dtype=np.int32
        )

        return face, bbox

    return None, None


# =========================================================
# GET EMBEDDING
# =========================================================

def get_embedding(image):

    if image is None or image.size == 0:
        return None

    face = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    face = cv2.resize(
        face,
        (160, 160)
    )

    face = face.astype(
        "float32"
    ) / 255.0

    face = np.transpose(
        face,
        (2, 0, 1)
    )

    face = np.expand_dims(
        face,
        axis=0
    )

    output = rec_model.run(
        None,
        {
            rec_model.get_inputs()[0].name:
            face
        }
    )

    return output[0][0]


# =========================================================
# TRAIN
# =========================================================

def train(
    saved_face,
    employee_id,
    N_SAMPLE=30
):

    global cls_model
    global cls_path
    global rec_model
    global cls_ready

    print("\n============================")
    print("BẮT ĐẦU TRAIN")
    print("Employee ID:", employee_id)
    print("Số ảnh nhận được:", len(saved_face))
    print("============================")

    # =====================================================
    # QUAN TRỌNG:
    # tránh IndexError khi không thu được khuôn mặt
    # =====================================================

    if saved_face is None or len(saved_face) == 0:

        print(
            "❌ Không có khuôn mặt nào được thu thập."
        )

        return {
            "status": "error",
            "message":
            "Không thu được khuôn mặt. "
            "Vui lòng đứng gần camera và đăng ký lại."
        }

    images = list(saved_face)

    n = len(images)

    # Nếu thiếu ảnh → nhân bản ngẫu nhiên
    if n < N_SAMPLE:

        samples = random.choices(
            images,
            k=N_SAMPLE - n
        )

        images.extend(samples)

    # Nếu nhiều hơn 30 → lấy ngẫu nhiên 30
    elif n > N_SAMPLE:

        images = random.sample(
            images,
            N_SAMPLE
        )

    print(
        "Số ảnh dùng để train:",
        len(images)
    )

    # =====================================================
    # Tạo embedding ảnh mới trước
    # =====================================================

    new_embeddings = []

    for image in images:

        if image is None or image.size == 0:
            continue

        # Ảnh gốc
        base_face = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        base_face = cv2.resize(
            base_face,
            (160, 160)
        )

        augmented_faces = augmentate(
            base_face
        )

        all_images = (
            [base_face] +
            augmented_faces
        )

        for img in all_images:

            img = img.astype(
                "float32"
            ) / 255.0

            img = np.transpose(
                img,
                (2, 0, 1)
            )

            img = np.expand_dims(
                img,
                axis=0
            )

            output = rec_model.run(
                None,
                {
                    rec_model
                    .get_inputs()[0]
                    .name: img
                }
            )

            vector = output[0][0]

            new_embeddings.append(
                vector
            )

    # Nếu vẫn không tạo được embedding
    if len(new_embeddings) == 0:

        print(
            "❌ Không tạo được embedding."
        )

        return {
            "status": "error",
            "message":
            "Không trích xuất được đặc trưng khuôn mặt."
        }

    print(
        "Embedding mới:",
        len(new_embeddings)
    )

    # =====================================================
    # Chỉ xoá embedding cũ SAU KHI embedding mới hợp lệ
    # =====================================================

    session.query(
        Embedding
    ).filter_by(
        employee_id=str(employee_id)
    ).delete()

    session.commit()

    instances = []

    for vector in new_embeddings:

        embedding_bytes = (
            vector
            .astype(np.float32)
            .tobytes()
        )

        instance = Embedding(
            embedding=embedding_bytes,
            employee_id=str(employee_id)
        )

        instances.append(instance)

    session.add_all(instances)

    session.commit()

    print(
        "✅ Đã lưu embedding vào database"
    )

    # =====================================================
    # LOAD TOÀN BỘ EMBEDDING
    # =====================================================

    embeddings = []
    labels = []

    data = (
        session
        .query(Embedding)
        .all()
    )

    for item in data:

        vector = np.frombuffer(
            item.embedding,
            dtype=np.float32
        )

        embeddings.append(
            vector.tolist()
        )

        labels.append(
            str(item.employee_id)
        )

    print(
        "Tổng embedding:",
        len(embeddings)
    )

    print(
        "Labels:",
        set(labels)
    )

    if len(embeddings) == 0:

        cls_ready = False

        return {
            "status": "error",
            "message":
            "Database không có embedding."
        }

    # =====================================================
    # Nếu chỉ có 1 nhân viên
    # SVC cần ít nhất 2 class
    # =====================================================

    unique_labels = set(labels)

    if len(unique_labels) < 2:

        embedding_size = len(
            embeddings[0]
        )

        # Dummy class unknown
        # dùng 2 sample vì cv=2
        dummy1 = np.random.normal(
            0,
            1,
            embedding_size
        ).tolist()

        dummy2 = np.random.normal(
            0,
            1,
            embedding_size
        ).tolist()

        embeddings.extend([
            dummy1,
            dummy2
        ])

        labels.extend([
            "unknown",
            "unknown"
        ])

        print(
            "⚠️ Chỉ có 1 nhân viên."
            " Đã thêm class unknown."
        )

    # =====================================================
    # TRAIN SVC
    # =====================================================

    try:

        base_model = LinearSVC(
            random_state=42
        )

        cls_model = (
            CalibratedClassifierCV(
                base_model,
                cv=2
            )
        )

        cls_model.fit(
            embeddings,
            labels
        )

        joblib.dump(
            cls_model,
            cls_path
        )

        cls_ready = True

        print(
            "✅ TRAIN THÀNH CÔNG"
        )

        print(
            "Classes:",
            cls_model.classes_
        )

        print(
            "Model:",
            cls_path
        )

        return {
            "status": "success",
            "message":
            "Huấn luyện khuôn mặt thành công"
        }

    except Exception as e:

        cls_ready = False

        print(
            "❌ TRAIN ERROR:",
            e
        )

        return {
            "status": "error",
            "message": str(e)
        }


# =========================================================
# REFRESH TRAIN
# =========================================================

def refresh_train(employee_id):

    global cls_model
    global cls_path
    global cls_ready

    # Xóa embedding nhân viên
    session.query(
        Embedding
    ).filter_by(
        employee_id=str(employee_id)
    ).delete()

    session.commit()

    embeddings = []
    labels = []

    data = (
        session
        .query(Embedding)
        .all()
    )

    for item in data:

        vector = np.frombuffer(
            item.embedding,
            dtype=np.float32
        )

        embeddings.append(
            vector.tolist()
        )

        labels.append(
            str(item.employee_id)
        )

    # Không còn nhân viên
    if len(embeddings) == 0:

        print(
            "⚠️ Không còn embedding."
        )

        cls_model = None
        cls_ready = False

        if os.path.exists(cls_path):

            try:
                os.remove(cls_path)
            except Exception:
                pass

        return {
            "status": "success",
            "message":
            "Không còn dữ liệu khuôn mặt."
        }

    if len(set(labels)) < 2:

        embedding_size = len(
            embeddings[0]
        )

        dummy1 = np.random.normal(
            0,
            1,
            embedding_size
        ).tolist()

        dummy2 = np.random.normal(
            0,
            1,
            embedding_size
        ).tolist()

        embeddings.extend([
            dummy1,
            dummy2
        ])

        labels.extend([
            "unknown",
            "unknown"
        ])

    base_model = LinearSVC(
        random_state=42
    )

    cls_model = (
        CalibratedClassifierCV(
            base_model,
            cv=2
        )
    )

    cls_model.fit(
        embeddings,
        labels
    )

    joblib.dump(
        cls_model,
        cls_path
    )

    cls_ready = True

    print(
        "✅ Refresh SVC thành công"
    )

    print(
        "Classes:",
        cls_model.classes_
    )

    return {
        "status": "success"
    }


# =========================================================
# PREDICT
# =========================================================

def predict(
    face,
    threshold=0.6
):

    global cls_model
    global rec_model
    global cls_ready

    # Chưa train
    if (
        not cls_ready
        or cls_model is None
    ):

        print(
            "⚠️ SVC chưa sẵn sàng"
        )

        return ""

    if face is None or face.size == 0:

        return ""

    try:

        img = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (160, 160)
        )

        img = img.astype(
            "float32"
        ) / 255.0

        img = np.transpose(
            img,
            (2, 0, 1)
        )

        img = np.expand_dims(
            img,
            axis=0
        )

        outputs = rec_model.run(
            None,
            {
                rec_model
                .get_inputs()[0]
                .name: img
            }
        )

        predicted_probs = (
            cls_model
            .predict_proba(
                outputs[0]
            )
        )

        confidence = float(
            np.max(
                predicted_probs
            )
        )

        predicted_index = int(
            np.argmax(
                predicted_probs,
                axis=1
            )[0]
        )

        label = (
            cls_model
            .classes_[
                predicted_index
            ]
        )

        # ============================
        # DEBUG
        # ============================

        print(
            f"FACE -> "
            f"label={label} | "
            f"confidence={confidence:.3f}"
        )

        # Không chấp nhận unknown
        if str(label) == "unknown":

            return ""

        if confidence >= threshold:

            print(
                f"✅ NHẬN DIỆN: "
                f"{label} "
                f"({confidence:.2%})"
            )

            return str(label)

        print(
            f"❌ Confidence thấp: "
            f"{confidence:.2%}"
        )

        return ""

    except Exception as e:

        print(
            "❌ Predict error:",
            e
        )

        return ""