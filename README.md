# Real-Life-Controls
Control mouse and keyboard through real life poses


# Product Demo

https://github.com/matt-wats/Real-Life-Controls/assets/112960646/edb233b0-7087-409d-b23d-b0749b8f0d97



# Motivation
Things I like: Playing Video Games

Things I don't Like: Sitting for a long time

Idea: Way to play video games while standing with what technology I already have (camera)

# Project Description
The camera tracks the poses my body is making and the gestures I am making with my hands, in order to control inputs to the computer.
I created the program to work specifically for playing Minecraft, but due to the nature of the program this can be easily adapted to many games/tasks.

The program looks at which way a person's body is leaning, where their hands (wrists) are and what gesture they're making, and how high/low they are.
In Minecraft, these actions map to the following controls:

**Basic Movement**

1. Directional Movement (WASD)
2. Jump & Crouch

**Mouse**

3. Left/Right Click
4. Looking Around (turning)

**Extra Inputs**

5. Change Hotbar Selection
6. Change Perspective
7. Drop Item
8. Open/Close Inventory


# How to use the Program

1. Configuring Controls

There is a dictionary that maps each real world action to a computer control. It can control either the keyboard, mouse clicking, or scroll wheel. You can adjust what everything does, with the notable exception of your right wrist moving the mouse.


2. Learning Phase

The program starts by "learning" what your body looks like doing certain actions.
In order it:
- Sets the center of your right wrist, and height of your hips
- Learns what your hand looks like while open, then closed (move your hands around while holding the gesture to improve it's accuracy)
- Learns what your body looks like while standing upright, leaning forward, then leaning backward (tilt side to side, squat, and jump while leaning to improve it's accuracy)


3. Action Phase

Now that the program has learned what certain actions look like, it can properly control the computer. Perform any real world actions that have been configured to control the computer. 
Keep your right hand down (near your hips), to pause the program from updating controls.



# How the Program Works

## Body Controls

1. Side to Side

This was relatively easy to implement. My chosen implmentation was the following:
If you want to move right, your left shoulder must be farther right than your left hip by some predefined distance. (and vice versa for moving left)

2. Forward and Backward

This was perhaps the most interesting control that I implemented. When you lean forward vs backward, your shoulders will get lower than when you're standing upright, but there's no clear way
to determine which, if any, direction you are tipping forward vs backward from should height alone.

MY IDEA: Use the brightness of the user's shirt to help make this decision.
EXPLANATION: In my case, the main light source in the room is from a ceiling light, so when I lean forward and my chest is pointed away from it, my shirt is darker. When I lean backward, with
my chest toward the light, my shirt is brighter.

So for testing which data was most important and which models predicted the lean best, I looked at variables such as: Shoulder Y, Hip Y, Tilt (side to side), Distance from Hips to Shoulders,
Mean Brightness of Shirt, Variation of Brightness of Shirt, etc.

I found the best model to be a Random Forest, using the Mean and Variation of Shirt Brightness, Hips Y, and Distance from Hips to Shoulders.

3. Jumping and Crouching

Jump in the air to jump, and squat to crouch. In more techincal terms: to jump, your hips must be above a certain height, and to crouch, your hips must be below a certain height (both can be adjusted).


## Right Hand Controls

1. Looking around (moving the mouse)

At the beginning of the program, the user holds their right hand up to "center" the wrist. To move the mouse, you move your wrist and the direction and speed is chosen relative to the center
e.g. To move the mouse down, you hold your wrist below the center. The farther away from the center your wrist is, the faster the mouse moves.

Here is a plot of how the mouse moves vs. how far your wrist is from the center (the red line and slope are adjustable)
![Mouse Movement Graph](https://github.com/matt-wats/Real-Life-Controls/assets/112960646/8fd28af3-c4f1-4681-969c-cd3b4d256739)


3. Clicking

At the beginning of the program, the user has to hold their hand open in different positions and then closed in different directions. Then to left and right click, you open your right hand with your fingers above and below your wrist, respectively.

To tell whether a hand it open or closed, I used the (x,y,z) coordinates of the wrist and tips of each finger. 
After testing I found that the distance of each finger to the one closest to it (and thumb to pinky) was the best input for models to decipher the hands' gestures, and the best model for this I found to be K-Nearest Neighbors.


## Left Hand Controls

Much like with your right hand, you open it to perform an action. You open your left hand above or below your wrist for two different functions, as well as above or below your shoulder for a different set of functions.
This works like the table below:

|               | Open Hand Upwards  | Open Hand Downwards |
| ------------- | ------------- | ------------- |
| **Above Shoulder**  | Action A  | Action B  |
| **Below Shoulder**  | Action C  | Action D  |

For me the controls are:
- Action A: Change Perspective (Q)
- Action B: Drop Item (G)
- Action C: Scroll Hotbar (Scroll Wheel -1)
- Action D: Open/Close Inventory (E)

I am aware that I have *unique* keybinds for minecraft.

# Tools Used
I used two models in this project: the yolov8 pose detection model (https://github.com/ultralytics/ultralytics), and the google mediapipe hand detection model (https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).

For predictions of actions from gestures, I looked at the following models: K-Neareset Neighbors, Decision Trees, Random Forests, Support Vector Machines (Linear and Polynomial)
