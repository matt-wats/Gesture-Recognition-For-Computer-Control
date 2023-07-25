
from ultralytics.yolo.engine.results import Results
import numpy as np
import torch
import cv2

keypoint2index = {
    "nose": 0,
    "left eye": 1,
    "right eye": 2,
    "left ear": 3,
    "right ear": 4,
    "left shoulder": 5,
    "right shoulder": 6,
    "left elbow": 7,
    "right elbow": 8,
    "left wrist": 9,
    "right wrist": 10,
    "left hip": 11,
    "right hip": 12,
    "left knee": 13,
    "right knee": 14,
    "left ankle": 15,
    "right ankle": 16,
}
keypart2index = {
    "nose": 0,
    "eyes": [1,2],
    "ears": [3, 4],
    "shoulders": [5, 6],
    "elbows": [7, 8],
    "wrists": [9, 10],
    "hips": [11, 12],
    "knees": [13, 14],
    "ankles": [15, 16],
}
CONF_THRESHOLD = 0.8

def interpret_result(result: Results) -> tuple: # (Bool, np.ndarray, torch.Tensor, torch.Tensor)
    img = result.orig_img
    keypoints = result.keypoints

    if keypoints.conf is None:
        return False, img, None, None

    conf = keypoints.conf[0]
    xy = keypoints.xy[0]

    return True, img, conf, xy

def get_upper_body_values(img: np.ndarray, xy: torch.Tensor, conf: torch.Tensor) -> tuple:

    point_indices = [
        keypoint2index["left shoulder"], 
        keypoint2index["right shoulder"], 
        keypoint2index["right hip"], 
        keypoint2index["left hip"],
    ]

    if conf[point_indices].min() < CONF_THRESHOLD:
        return False, None

    points = xy[[point_indices]]
    points = points.cpu().int().numpy()
    hips_y = points[2:4,1].mean()

    tilt = points[0:2, 0].mean() - points[2:4, 0].mean()
    tip = points[0:2, 1].mean() - points[2:4, 1].mean()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(gray_image)
    mask = cv2.fillPoly(mask, [points], (255))
    selected_pixel_values = gray_image[mask > 0]

    values = [selected_pixel_values.mean(), selected_pixel_values.std(), tip, hips_y]

    return True, values