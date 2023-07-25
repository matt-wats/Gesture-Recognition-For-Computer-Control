import numpy as np
import cv2
import mediapipe as mp

from torch import Tensor

from result_functions import keypoint2index, CONF_THRESHOLD, keypart2index
from zone_functions import get_is_active

import mouse


def do_hand_controls(img: np.ndarray, xy: Tensor, conf: Tensor, 
                     centers: dict, senses: dict, detectors: dict,
                     actions: dict) -> bool:
    
    hand_results = get_hand_results(img=img, detector=detectors["hands"])
    
    movement_worked = do_mouse_movement(xy=xy, conf=conf, centers=centers, senses=senses)
    click_worked = do_mouse_click(right_results=hand_results["Right"], detector=detectors["right gesture"], actions=actions)
    left_worked = do_left_hand(left_results=hand_results["Left"], xy=xy, conf=conf, centers=centers, actions=actions, senses=senses, detector=detectors["left gesture"])

    return True


# -------------------------------------------------------------------------------------------------
# RIGHT HAND MOUSE CLICK
# -------------------------------------------------------------------------------------------------
def do_mouse_click(right_results: list, detector, actions: dict) -> bool:
    if not right_results["worked"]:
        return False
    
    click_type = "left" if right_results["type"] == "primary" else "right"
    
    cls = detector.predict(values=right_results["values"])
    if cls == "open":
        actions["states"][f"click {click_type}"][0] = True
    return True



# -------------------------------------------------------------------------------------------------
# HAND AND FINGER VALUE DETECTION
# -------------------------------------------------------------------------------------------------
def get_3d_distance(a: list, b: list) -> float:
    x = a[0]-b[0]
    y = a[1]-b[1]
    z = a[2]-b[2]
    return x**2 + y**2 + z**2


def get_hand_results(img: np.ndarray, detector) -> tuple:
    cvt_img = cv2.cvtColor(np.fliplr(img), cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(data=cvt_img, image_format=mp.ImageFormat.SRGB)
    detection_result = detector.detect(mp_img)

    hands = ["Left", "Right"]
    outputs = dict((hand, {"worked": False}) for hand in hands)

    for hand in hands:
        try:
            hand_idx = [handedness[0].display_name for handedness in detection_result.handedness].index(hand)
        except:
            continue
        points = detection_result.hand_landmarks[hand_idx][::4]

        wrist_xyz = [points[0].x, points[0].y, points[0].z]
        fingers_xyz = [[point.x, point.y, point.z] for point in points[1:]]

        activation_type = "primary" if np.array(fingers_xyz)[:,1].mean() <= wrist_xyz[1] else "secondary"
        wrist_distances = [get_3d_distance(wrist_xyz, finger_xyz) for finger_xyz in fingers_xyz]
        fingers_distances = [get_3d_distance(fingers_xyz[i], fingers_xyz[i+1]) for i in range(-1,len(fingers_xyz)-1)]
        fingers_values = fingers_distances#[*wrist_distances, *fingers_distances]

        outputs[hand]["worked"] = True
        outputs[hand]["values"] = fingers_values
        outputs[hand]["type"] = activation_type

    return outputs




# -------------------------------------------------------------------------------------------------
# LEFT HAND CONTROLS
# -------------------------------------------------------------------------------------------------
def do_left_hand(left_results, xy: Tensor, conf: Tensor, centers: dict, actions: dict, senses: dict, detector) -> bool:

    if not left_results["worked"]:
        return False
    
    active = get_is_active(xy=xy, conf=conf, senses=senses, hand="left")
    if not active:
        return False

    left_wrist_idx = keypoint2index["left wrist"]
    
    cls = detector.predict(values=left_results["values"])
    if cls == "close":
        return True    

    left_wrist_y = xy[left_wrist_idx][1]
    activation_type = left_results["type"]

    shoulders_y = centers["SHOULDERS Y"]
    shoulders_idx = keypart2index["shoulders"]
    if conf[shoulders_idx].min() > CONF_THRESHOLD:
        shoulders_y = xy[shoulders_idx][:,1].mean()

    if left_wrist_y < shoulders_y:
        actions["states"][f"above {activation_type}"][0] = True
    if left_wrist_y > shoulders_y:
        actions["states"][f"below {activation_type}"][0] = True

    return True



# -------------------------------------------------------------------------------------------------
# RIGHT HAND MOUSE MOVEMENT CONTROLS
# -------------------------------------------------------------------------------------------------
def do_mouse_movement(xy, conf, centers: dict, senses: dict) -> bool:
    r_wrist_idx = keypoint2index["right wrist"]
    if conf[r_wrist_idx] < CONF_THRESHOLD:
        return False
    r_wrist_pos = xy[r_wrist_idx]
    move_x, move_y = get_mouse_movement(r_wrist_pos, centers["R WRIST"], senses)

    mouse.move(move_x, move_y, absolute=False)
    return True


def get_mouse_movement(curr_right_wrist, center_right_wrist, senses: dict) -> tuple:
    dist_x = (center_right_wrist[0] - curr_right_wrist[0]).item()
    dist_y = (curr_right_wrist[1] - center_right_wrist[1]).item()

    if dist_x > senses["MOUSE ACTIVATION"]:
        move_x = senses["MOUSE X"] * (dist_x - senses["MOUSE ACTIVATION"])
    elif dist_x < -senses["MOUSE ACTIVATION"]:
        move_x = senses["MOUSE X"] * (dist_x + senses["MOUSE ACTIVATION"])
    else:
        move_x = 0

    if dist_y > senses["MOUSE ACTIVATION"]:
        move_y = senses["MOUSE Y"] * (dist_y - senses["MOUSE ACTIVATION"])
    elif dist_y < -senses["MOUSE ACTIVATION"]:
        move_y = senses["MOUSE Y"] * (dist_y + senses["MOUSE ACTIVATION"])
    else:
        move_y = 0
    
    return move_x, move_y