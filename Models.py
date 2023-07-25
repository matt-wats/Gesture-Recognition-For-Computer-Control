from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from ultralytics import YOLO

from result_functions import CONF_THRESHOLD


def get_hand_detection_model():
    model_path = "hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options,
                                        num_hands=2)
    hand_detector = vision.HandLandmarker.create_from_options(options)

    return hand_detector

def get_yolo_model() -> tuple:
    model = YOLO("yolov8n-pose.pt", task="pose")
    results = model.predict(source="0", show=False, stream=True, verbose=False, conf=CONF_THRESHOLD)

    return model, results