from torch import Tensor
from result_functions import keypoint2index, keypart2index, CONF_THRESHOLD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def locate_centers(xy: Tensor, conf: Tensor) -> tuple:

    important_indices = [
        keypoint2index["right wrist"], 
        *keypart2index["shoulders"], 
        *keypart2index["hips"]
        ]
    
    if conf[important_indices].min() < CONF_THRESHOLD:
        return False, None, None, None

    r_wrist_center = xy[keypoint2index["right wrist"]].tolist()

    shoulder_y = xy[keypart2index["shoulders"]][:,1].mean().item()
    hip_y = xy[keypart2index["hips"]][:,1].mean().item()

    return True, r_wrist_center, shoulder_y, hip_y


def get_is_active(xy: Tensor, conf: Tensor, senses: dict, hand: str = "right") -> bool:

    wrist_idx = keypoint2index[f"{hand} wrist"]
    hips_indicies = keypart2index["hips"]
    hips_y = xy[hips_indicies][:,1].mean()

    if conf[[wrist_idx, *hips_indicies]].min() < CONF_THRESHOLD:
        return False

    if xy[wrist_idx][1] + senses["ACTIVE"] > hips_y:
        return False

    return True





class LeanDetector():
    def __init__(self) -> None:
        self.stored_lean_values = []
        self.stored_lean_classes = []

        self.classifier = RandomForestClassifier(max_depth=3, random_state=42, class_weight={0:5, 1:2, 2:1}, n_estimators=100)

        self.lean_classes = ["up", "forward", "backward"]
        self.lean2class = dict(zip(self.lean_classes, np.arange(0, len(self.lean_classes))))

    def add_lean(self, values: list, lean: str) -> None:
        self.stored_lean_values.append(values)
        self.stored_lean_classes.append(self.lean2class[lean])

    def fit_classifier(self) -> None:
        self.classifier.fit(self.stored_lean_values, self.stored_lean_classes)

    def predict(self, values: list) -> str:
        class_idx = self.classifier.predict([values])[0]
        return self.lean_classes[class_idx]


class HandGesture():
    def __init__(self, k: int = 31, class_0_threshold: float = 2/3) -> None:
        self.stored_gesture_values = []
        self.stored_gesture_classes = []

        self.classifer = KNeighborsClassifier(n_neighbors=k)

        self.class_0_threshold = class_0_threshold
        self.gesture_classes = ["open", "close"]
        self.gesture2class = dict(zip(self.gesture_classes, np.arange(0,len(self.gesture_classes))))


    def add_gesture(self, values: list, gesture: str) -> None:
        self.stored_gesture_values.append(values)
        self.stored_gesture_classes.append(self.gesture2class[gesture])

    def fit_classifer(self) -> None:
        self.classifer.fit(self.stored_gesture_values, self.stored_gesture_classes)
        

    def predict(self, values: list) -> str:
        class_idx = int(self.classifer.predict_proba([values])[0][0] < self.class_0_threshold)
        return self.gesture_classes[class_idx]
    