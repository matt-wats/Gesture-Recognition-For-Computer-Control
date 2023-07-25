import zone_functions
import result_functions
import Models

import Hand_Controls
import Body_Controls
import Information_Interactions

# ----------------------------------------------------------------------------------------------------------------------------------
# Display Window
# ----------------------------------------------------------------------------------------------------------------------------------

display_window = Information_Interactions.DisplayWindow()


# ----------------------------------------------------------------------------------------------------------------------------------
# Defining Values
# ----------------------------------------------------------------------------------------------------------------------------------

# Sensitivites for actions
SENSES = {
    "MOUSE X": 2.0, # Wrist distance in camera pixels -> screen pixel distance
    "MOUSE Y": 2.0, # Wrist distance in camera pixels -> screen pixel distance
    "MOUSE ACTIVATION": 15, # Wrist distance in camera pixels needed to move
    "JUMP": 50, # Hip height in pixels above needed for jump
    "CROUCH": 50, # Hip height in pixels below needed for crouch
    "TILT": 50, # Shoulder past hip distance in pixels for lean
    "ACTIVE": 20, # How far above in pixels your hip your hand can be to activate
}

# Mapping action to input controls
action_controls = {
    "forward": ("key", "w"),
    "backward": ("key", "s"),
    "left": ("key", "a"),
    "right": ("key", "d"),
    "jump": ("key", " "),
    "crouch": ("key", "left shift"),

    "above primary": ("key", "q"),
    "below secondary": ("key", "e"),
    "above secondary": ("key", "g"),
    "below primary": ("wheel", -1),

    "click left": ("click", "left"),
    "click right": ("click", "right"),
}
actions = Information_Interactions.action_controls2actions(action_controls=action_controls)

# How long each section lasts
CENTER_COUNTDOWN = 120
GESTURE_COUNTDOWN = 90
LEAN_COUNTDOWN = 120
WAIT_COUNTDOWN = 60


# ----------------------------------------------------------------------------------------------------------------------------------
# Loading Models
# ----------------------------------------------------------------------------------------------------------------------------------

# YOLO MODEL
model, results = Models.get_yolo_model()

# Detectors: hand detection, hand gestures, lean detection
detectors = {
    "hands": Models.get_hand_detection_model(),
    "right gesture": zone_functions.HandGesture(k=1+GESTURE_COUNTDOWN//2),
    "left gesture": zone_functions.HandGesture(k=1+GESTURE_COUNTDOWN//2),
    "lean": zone_functions.LeanDetector(),
}


# ----------------------------------------------------------------------------------------------------------------------------------
# Center Loop
# ----------------------------------------------------------------------------------------------------------------------------------

# values of "center"/"origin" for certain body keypoints
centers = {
    "R WRIST": None,
    "HIPS Y": None,
    "SHOULDERS Y": None,
}

countdown = CENTER_COUNTDOWN
# Loop for Locating centers
for result in results:
    display_window.update_label(Information_Interactions.learning_info2str("Locating centers", "NA", countdown))
    Information_Interactions.show_results(result.plot(boxes=False))

    worked, img, conf, xy = result_functions.interpret_result(result)
    if not worked: continue
    active = zone_functions.get_is_active(xy, conf, SENSES)
    if not active:
        continue

    countdown -= 1
    if countdown > 0:
        continue

    worked, wrist, shoulder, hip = zone_functions.locate_centers(xy, conf)
    if worked:
        centers["R WRIST"] = wrist
        centers["HIPS Y"] = hip
        centers["SHOULDERS Y"] = shoulder

        break
    else:
        countdown = CENTER_COUNTDOWN


# ----------------------------------------------------------------------------------------------------------------------------------
# Gesture Loop
# ----------------------------------------------------------------------------------------------------------------------------------

countdown = GESTURE_COUNTDOWN
wait = WAIT_COUNTDOWN
gesture_type = "open"
# Loop of Learning Hand Gestures
for result in results:
    display_window.update_label(Information_Interactions.learning_info2str("Learning Hand Gestures", gesture_type, countdown, wait))
    Information_Interactions.show_results(result.plot(boxes=False), centers=centers, senses=SENSES)

    if wait > 0:
        wait -= 1
        continue
    worked, img, conf, xy = result_functions.interpret_result(result)
    if not worked: continue

    hand_results = Hand_Controls.get_hand_results(img=img, detector=detectors["hands"])
    if hand_results["Left"]["worked"] and hand_results["Right"]["worked"]:
        detectors["right gesture"].add_gesture(values=hand_results["Right"]["values"], gesture=gesture_type)
        detectors["left gesture"].add_gesture(values=hand_results["Left"]["values"], gesture=gesture_type)
    else:
        continue

    countdown -= 1
    if countdown == 0:
        if gesture_type == "open":
            gesture_type = "close"
            countdown = GESTURE_COUNTDOWN
            wait = WAIT_COUNTDOWN
        elif gesture_type == "close":
            break
detectors["right gesture"].fit_classifer()
detectors["left gesture"].fit_classifer()


# ----------------------------------------------------------------------------------------------------------------------------------
# Lean Loop
# ----------------------------------------------------------------------------------------------------------------------------------

countdown = LEAN_COUNTDOWN
wait = WAIT_COUNTDOWN
lean_type = "up"
# Loop for Learning Leans
for result in results:
    display_window.update_label(Information_Interactions.learning_info2str("Learning Leans", lean_type, countdown, wait))
    Information_Interactions.show_results(result.plot(boxes=False), centers=centers, senses=SENSES)

    if wait > 0:
        wait -= 1
        continue
    worked, img, conf, xy = result_functions.interpret_result(result)
    if not worked: continue

    worked, upper_body_values = result_functions.get_upper_body_values(img, xy, conf)
    if worked:
        detectors["lean"].add_lean(upper_body_values, lean_type)
    else:
        continue

    countdown -= 1
    if countdown == 0:
        if lean_type == "up":
            lean_type = "forward"
            countdown = LEAN_COUNTDOWN
            wait = WAIT_COUNTDOWN
        elif lean_type == "forward":
            lean_type = "backward"
            countdown = LEAN_COUNTDOWN
            wait = WAIT_COUNTDOWN
        elif lean_type == "backward":
            break
detectors["lean"].fit_classifier()



# ----------------------------------------------------------------------------------------------------------------------------------
# Action Loop
# ----------------------------------------------------------------------------------------------------------------------------------

# Loop for controlling computer
for result in results:

    Information_Interactions.step_states(actions=actions)
    Information_Interactions.show_results(result.plot(boxes=False), centers=centers, senses=SENSES)

    worked, img, conf, xy = result_functions.interpret_result(result)
    if not worked: continue

    active = zone_functions.get_is_active(xy, conf, SENSES)
    if not active:
        continue

    output = Hand_Controls.do_hand_controls(img=img, xy=xy, conf=conf, centers=centers, senses=SENSES, detectors=detectors, actions=actions)
    output = Body_Controls.do_body_controls(img=img, conf=conf, centers=centers, senses=SENSES, detectors=detectors, actions=actions, xy=xy)

    Information_Interactions.perform_actions(actions=actions)
    display_window.update_label(Information_Interactions.states2str(actions["states"]))


# Run the main window loop
display_window.window.mainloop()