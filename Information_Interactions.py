import tkinter as tk
from tkinter import font

import keyboard
import mouse

import numpy as np
import cv2

# ----------------------------------------------------------------------------------------------------------------------------------
# DOING ACTIONS
# ----------------------------------------------------------------------------------------------------------------------------------
def do_control(control_type: str, bind, undo: bool = False) -> None:
    if control_type == "key":
        if undo:
            keyboard.release(bind)
        else:
            keyboard.press(bind)
    elif control_type == "click":
        if undo:
            if mouse.is_pressed(bind): mouse.release(bind)
        else:
            mouse.press(bind)
    elif control_type == "wheel":
        if undo:
            pass
        else:
            mouse.wheel(delta=bind)

def perform_actions(actions: dict) -> None:
    for action,state in actions["states"].items():
        if not state[0]:
            do_control(*actions["controls"][action], undo=True)
        elif state[0] and (not state[1]):
            do_control(*actions["controls"][action])


# ----------------------------------------------------------------------------------------------------------------------------------
# DISPAY/PLOT DETECTION RESULTS
# ----------------------------------------------------------------------------------------------------------------------------------

def show_results(img: np.array, centers: dict = None, senses: dict = None) -> None:
    winname = "Detection Results"

    height,width,channels = img.shape

    thickness = 3

    if centers == None:
        cv2.imshow(winname=winname, mat=np.fliplr(img))
        return None
    
    CENTER_COLOR = (255,0,0)
    BOUND_COLOR = (0,255,0)
    ACTIVE_COLOR = (0,0,255)

    # wrist box
    x,y = centers["R WRIST"]
    pt1 = (int(x-senses["MOUSE ACTIVATION"]), int(y-senses["MOUSE ACTIVATION"]))
    pt2 = (int(x+senses["MOUSE ACTIVATION"]), int(y+senses["MOUSE ACTIVATION"]))
    cv2.rectangle(img, pt1, pt2, color=BOUND_COLOR, thickness=thickness)


    # hips lines
    y = centers["HIPS Y"]
    cv2.line(img, (0, int((y-senses["JUMP"]))), (width, int(y-senses["JUMP"])), color=BOUND_COLOR, thickness=thickness)
    cv2.line(img, (0, int(y+senses["CROUCH"])), (width, int(y+senses["CROUCH"])), color=BOUND_COLOR, thickness=thickness)

    # active line
    y = centers["HIPS Y"]
    cv2.line(img, (0, int((y-senses["ACTIVE"]))), (width, int(y-senses["ACTIVE"])), color=ACTIVE_COLOR, thickness=thickness)

    # shoulders line
    y = centers["SHOULDERS Y"]
    cv2.line(img, (0, int(y)), (width, int(y)), color=CENTER_COLOR, thickness=thickness)


    cv2.imshow(winname=winname, mat=np.fliplr(img))
    cv2.waitKey(1)



# ----------------------------------------------------------------------------------------------------------------------------------
# CREATING ACTIONS DICT FROM CONTROLS
# ----------------------------------------------------------------------------------------------------------------------------------
def action_controls2actions(action_controls: dict) -> dict:
    action_states = dict((key, 2*[False]) for key in action_controls.keys())

    actions = {
        "controls": action_controls,
        "states": action_states,
    }

    return actions

# ----------------------------------------------------------------------------------------------------------------------------------
# RESET STATES FOR EACH LOOP
# ----------------------------------------------------------------------------------------------------------------------------------
def step_states(actions: dict) -> None:
    for key in actions["states"].keys():
        actions["states"][key][1] = actions["states"][key][0]
        actions["states"][key][0] = False


# ----------------------------------------------------------------------------------------------------------------------------------
# Creating output string for a given dict
# ----------------------------------------------------------------------------------------------------------------------------------
def states2str(action_states: dict) -> str:
    s = "ACTIVE CONTROLS:\n"
    for action,state in action_states.items():
        s += f"{action}: {state[0]}\n"
    return s


def learning_info2str(learning_mode: str, mode_type: str, count: int, wait: int = 0) -> str:
    s = f"Learning Mode: {learning_mode}\n"
    s += f"Mode Type: {mode_type}\n"
    s += f"Countdown: {count}\n"
    if wait > 0:
        s += f"Wait Delay: {wait}"
    return s

# ----------------------------------------------------------------------------------------------------------------------------------
# Display Window class
# ----------------------------------------------------------------------------------------------------------------------------------
class DisplayWindow():
    def __init__(self) -> None:
        self.window, self.label = self.create_window()

    def create_window(self) -> tuple:
        # Create the main window
        window = tk.Tk()
        label = tk.Label(window, 
                        text="Starting up...", 
                        font=font.Font(size=30),
                        background="lime",
                        )
        label.pack()
        label.update()
        return window, label
    
    def update_label(self, text: str) -> None:
        self.label.config(text=text)
        self.label.update()