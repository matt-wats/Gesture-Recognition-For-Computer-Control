import numpy as np
from torch import Tensor

import result_functions
import zone_functions

def do_body_controls(img: np.ndarray, conf: Tensor, 
                     centers: dict, senses: dict, 
                     detectors, actions: dict, xy: Tensor) -> bool:
    
    crouch_worked = do_crouch(xy=xy, conf=conf, senses=senses, centers=centers, actions=actions)
    side_worked = do_side(xy=xy, conf=conf, senses=senses, actions=actions)
    walk_worked = do_walk(img=img, xy=xy, conf=conf, detector=detectors["lean"], actions=actions)
    jump_worked = do_jump(xy=xy, conf=conf, senses=senses, centers=centers, actions=actions)

    return True



# -------------------------------------------------------------------------------------------------
# CROUCHING
# -------------------------------------------------------------------------------------------------
def do_crouch(xy: Tensor, conf: Tensor, senses: dict, centers: dict, actions: dict) -> bool:
    hips_indices = result_functions.keypart2index["hips"]

    if conf[hips_indices].min() < result_functions.CONF_THRESHOLD:
        return False
    hips_y = xy[hips_indices][:,1].mean().item()
    if hips_y - senses["CROUCH"] > centers["HIPS Y"]:
        actions["states"]["crouch"][0] = True
    return True

# -------------------------------------------------------------------------------------------------
# CROUCHING
# -------------------------------------------------------------------------------------------------
def do_jump(xy: Tensor, conf: Tensor, senses: dict, centers: dict, actions: dict) -> bool:

    hips_indices = result_functions.keypart2index["hips"]

    if conf[hips_indices].min() < result_functions.CONF_THRESHOLD:
        return False
    hips_y = xy[hips_indices][:,1].mean().item()
    if hips_y + senses["JUMP"] < centers["HIPS Y"]:
        actions["states"]["jump"][0] = True
    return True

# -------------------------------------------------------------------------------------------------
# CROUCHING
# -------------------------------------------------------------------------------------------------
def do_side(xy: Tensor, conf: Tensor, senses: dict, actions: dict) -> None:

    # LEFT X > RIGHT X

    # tilt left
    # if right shoulder is left of right hip (include sense) then go left
    r_shoulder_idx = result_functions.keypoint2index["right shoulder"]
    r_hip_idx = result_functions.keypoint2index["right hip"]
    if conf[[r_shoulder_idx, r_hip_idx]].min() > result_functions.CONF_THRESHOLD:
        if xy[r_shoulder_idx][0] - senses["TILT"] > xy[r_hip_idx][0]:
            actions["states"]["left"][0] = True

    # tilt right
    l_shoulder_idx = result_functions.keypoint2index["left shoulder"]
    l_hip_idx = result_functions.keypoint2index["left hip"]
    if conf[[l_shoulder_idx, l_hip_idx]].min() > result_functions.CONF_THRESHOLD:
        if xy[l_shoulder_idx][0] + senses["TILT"] < xy[l_hip_idx][0]:
            actions["states"]["right"][0] = True

    return True

# -------------------------------------------------------------------------------------------------
# CROUCHING
# -------------------------------------------------------------------------------------------------
def do_walk(img: np.ndarray, xy: Tensor, conf: Tensor, detector: zone_functions.LeanDetector, actions: dict) -> bool:

    worked, values = result_functions.get_upper_body_values(img, xy, conf)
    if not worked:
        return False
    
    lean_class = detector.predict(values)

    if lean_class in actions["states"]:
        actions["states"][lean_class][0] = True

    return True