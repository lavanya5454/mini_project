import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from PIL import Image, ImageTk

# Initialize the main window
root = tk.Tk()
root.title("Volume and Brightness Control")
root.geometry("800x600")

# Audio Volume Control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange

# MediaPipe Hands Model
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Tkinter Canvas for displaying the video feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Volume Slider
volume_label = tk.Label(root, text="Volume")
volume_label.pack()
volume_slider = ttk.Scale(root, from_=minVol, to=maxVol, orient="horizontal")
volume_slider.set(maxVol / 2)  # Initial volume value
volume_slider.pack()

# Brightness Slider
brightness_label = tk.Label(root, text="Brightness")
brightness_label.pack()
brightness_slider = ttk.Scale(root, from_=0, to=100, orient="horizontal")
brightness_slider.set(50)  # Initial brightness value
brightness_slider.pack()

def update_brightness(value):
    sbc.set_brightness(int(value))

def update_volume(value):
    volume.SetMasterVolumeLevel(float(value), None)

brightness_slider.config(command=update_brightness)
volume_slider.config(command=update_volume)

# Function to process the video and track hand gestures
def process_video():
    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    processed = hands.process(frameRGB)

    left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed, draw, mpHands)
    
    # Adjust brightness with left hand
    if left_landmark_list:
        left_distance = get_distance(frame, left_landmark_list)
        b_level = np.interp(left_distance, [50, 220], [0, 100])
        brightness_slider.set(b_level)  # Update the brightness slider

    # Adjust volume with right hand
    if right_landmark_list:
        right_distance = get_distance(frame, right_landmark_list)
        vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
        volume_slider.set(vol)  # Update the volume slider

    # Convert the image for Tkinter display
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)
    
    # Update canvas
    canvas.create_image(0, 0, image=imgtk, anchor="nw")
    root.after(10, process_video)  # Call the function repeatedly

def get_left_right_landmarks(frame, processed, draw, mpHands):
    left_landmark_list = []
    right_landmark_list = []

    if processed.multi_hand_landmarks:
        for handlm in processed.multi_hand_landmarks:
            for idx, found_landmark in enumerate(handlm.landmark):
                height, width, _ = frame.shape
                x, y = int(found_landmark.x * width), int(found_landmark.y * height)
                if idx == 4 or idx == 8:
                    landmark = [idx, x, y]
                    if handlm == processed.multi_hand_landmarks[0]:
                        left_landmark_list.append(landmark)
                    elif handlm == processed.multi_hand_landmarks[1]:
                        right_landmark_list.append(landmark)

            draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)

    return left_landmark_list, right_landmark_list

def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return 0
    (x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), (landmark_list[1][1], landmark_list[1][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    L = hypot(x2 - x1, y2 - y1)
    return L

# Start processing video and Tkinter loop
process_video()
root.mainloop()

# Release the video capture when done
cap.release()
cv2.destroyAllWindows()
