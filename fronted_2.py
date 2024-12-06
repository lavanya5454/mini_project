import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

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
root.title("Volume and Brightness Control System")  # Set the window title

# Set window size
root.geometry("800x600")

# Set the font style for the title (bold and large font)
root.option_add("*Font", "Helvetica 16 bold")

# Initialize Audio Volume Control (ensure this part runs before using volume sliders)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, _ = volRange  # Initialize minVol and maxVol

# MediaPipe Hands Model Initialization
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2)

draw = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    print("Camera opened successfully.")

# Load background image (replace with your image path)
bg_image = Image.open("C:\\Users\\Admin\\Desktop\\volume_brightness\\pic.jpeg")
bg_image = bg_image.resize((800, 600), Image.Resampling.LANCZOS)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create background label
bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)  # Fill the entire window with the image

# Title Label (above camera feed)
title_label = tk.Label(root, text="Volume and Brightness Control System", fg="white", bg="#1e1e1e", font=("Helvetica", 24, "bold"))
title_label.pack(pady=20)

# Tkinter Canvas for displaying the video feed (camera screen)
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack(pady=20)

# Volume Slider with Label (below camera feed)
volume_label = tk.Label(root, text="Volume", fg="white", bg="#1e1e1e", font=("Helvetica", 18))
volume_label.pack(pady=5)
volume_slider = ttk.Scale(root, from_=minVol, to=maxVol, orient="horizontal", length=400)
volume_slider.set(maxVol / 2)  # Initial volume value
volume_slider.pack(pady=10)

# Brightness Slider with Label (below the volume slider)
brightness_label = tk.Label(root, text="Brightness", fg="white", bg="#1e1e1e", font=("Helvetica", 18))
brightness_label.pack(pady=5)
brightness_slider = ttk.Scale(root, from_=0, to=100, orient="horizontal", length=400)
brightness_slider.set(50)  # Initial brightness value
brightness_slider.pack(pady=10)

# Update brightness function
def update_brightness(value):
    sbc.set_brightness(int(value))

# Update volume function
def update_volume(value):
    volume.SetMasterVolumeLevel(float(value), None)

brightness_slider.config(command=update_brightness)
volume_slider.config(command=update_volume)

# Function to process the video and track hand gestures
def process_video():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        return

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    processed = hands.process(frameRGB)

    # Convert the frame for Tkinter display
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    # Update canvas with the current frame
    canvas.create_image(0, 0, image=imgtk, anchor="nw")

    # Update the image reference to avoid garbage collection
    canvas.imgtk = imgtk  # Keep a reference to avoid garbage collection

    # Call the function repeatedly to process the video feed
    root.after(10, process_video)

# Start processing video feed
process_video()

# Run the Tkinter event loop
root.mainloop()

# Release the video capture when done
cap.release()
cv2.destroyAllWindows()
