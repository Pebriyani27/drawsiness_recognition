import cv2
import mediapipe as mp
import pygame
import threading
import tkinter as tk
from tkinter import Label, Frame
from PIL import Image, ImageTk
import numpy as np
import os
from math import hypot
import urllib.request

# Inisialisasi alarm
pygame.mixer.init()
ALARM_SOUND = "kunti ketawa.mp3"  # Pastikan file ini ada di folder yang sama dalam folder ini
if os.path.exists(ALARM_SOUND):
    pygame.mixer.music.load(ALARM_SOUND)
else:
    print(f"[ERROR] File alarm '{ALARM_SOUND}' tidak ditemukan.")
    exit()

# Download robot GIF animasi jika belum ada
ROBOT_GIF_URL = "https://img1.picmix.com/output/stamp/normal/1/3/7/3/613731_f407d.gif" # URL animasi hantu
ROBOT_GIF_FILE = "hantu.gif"

# Sesuaikan bagian download file GIF
try:
    if not os.path.exists(ROBOT_GIF_FILE):
        print("Mengunduh GIF robot animasi...")
        urllib.request.urlretrieve(ROBOT_GIF_URL, ROBOT_GIF_FILE)
        print("Download selesai.")
    else:
        print("File GIF robot sudah ada.")
except Exception as e:
    print(f"[ERROR] Gagal mengunduh gambar robot: {e}")

# Load GIF dan debug
robot_frames = []
robot_frame_index = 0

def load_robot_gif():
    global robot_frames
    try:
        robot_img = Image.open(ROBOT_GIF_FILE)
        for frame_index in range(robot_img.n_frames):
            robot_img.seek(frame_index)
            frame = robot_img.copy().convert('RGBA')
            robot_frames.append(ImageTk.PhotoImage(frame))
    except Exception as e:
        print(f"[ERROR] Gagal memuat GIF robot: {e}")

def animate_robot():
    global robot_frame_index
    if ALARM_ON and robot_frames:
        frame = robot_frames[robot_frame_index]
        label_robot_image.configure(image=frame)
        label_robot_image.image = frame  # prevent garbage collection
        robot_frame_index = (robot_frame_index + 1) % len(robot_frames)
        label_robot_text.config(text="BANGUN WOIII!!")
        window.after(100, animate_robot)
    else:
        label_robot_image.configure(image='')
        label_robot_text.config(text='')

# Fungsi untuk memainkan alarm
def sound_alarm():
    while ALARM_ON:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()

# Fungsi hitung Eye Aspect Ratio (EAR) berdasarkan koordinat landmark
def eye_aspect_ratio(eye):
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    if C == 0:
        return 0
    ear = (A + B) / (2.0 * C)
    return ear

# MediaPipe face mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Landmark indexes mata kiri dan kanan pada MediaPipe face mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]  # 6 titik mata kiri
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # 6 titik mata kanan

# Threshold dan frame deteksi
EAR_THRESHOLD = 0.15
EAR_CONSEC_FRAMES = 20
COUNTER = 0
ALARM_ON = False

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Tidak dapat membuka webcam.")
    exit()

# Setup tkinter GUI
window = tk.Tk()
window.title("Deteksi Kantuk")
window.geometry("800x600")

frame_main = Frame(window)
frame_main.pack()

label_video = Label(frame_main)
label_video.grid(row=0, column=0, padx=10, pady=10)

frame_robot = Frame(frame_main)
frame_robot.grid(row=0, column=1, sticky="n", padx=10, pady=10)

label_robot_image = Label(frame_robot)
label_robot_text = Label(frame_robot, text="", font=("Helvetica", 24, "bold"), fg="red")
label_robot_image.pack()
label_robot_text.pack()

# Load robot GIF setelah GUI siap
load_robot_gif()

def update_frame():
    global COUNTER, ALARM_ON

    ret, frame = cap.read()
    if not ret or frame is None:
        window.after(10, update_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]
    results = face_mesh.process(frame_rgb)

    ear = 0
    if results.multi_face_landmarks:
        mesh_points = results.multi_face_landmarks[0].landmark

        left_eye = []
        right_eye = []

        for idx in LEFT_EYE:
            x = int(mesh_points[idx].x * img_w)
            y = int(mesh_points[idx].y * img_h)
            left_eye.append((x, y))

        for idx in RIGHT_EYE:
            x = int(mesh_points[idx].x * img_w)
            y = int(mesh_points[idx].y * img_h)
            right_eye.append((x, y))

        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        ear = (left_EAR + right_EAR) / 2.0

        cv2.polylines(frame_rgb, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 0), 1)
        cv2.polylines(frame_rgb, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 0), 1)

        cv2.putText(frame_rgb, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if ear < EAR_THRESHOLD:
            COUNTER += 1
            if COUNTER >= EAR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    threading.Thread(target=sound_alarm, daemon=True).start()
                    animate_robot()  # Start animating robot when alarm is on
                cv2.putText(frame_rgb, "BANGUN WOIII!!", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            COUNTER = 0
            if ALARM_ON:
                ALARM_ON = False
                pygame.mixer.music.stop()
    else:
        COUNTER = 0
        if ALARM_ON:
            ALARM_ON = False
            pygame.mixer.music.stop()

    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label_video.imgtk = imgtk  # type: ignore
    label_video.configure(image=imgtk)

    window.after(10, update_frame)

def on_closing():
    cap.release()
    pygame.mixer.quit()
    window.destroy()

update_frame()
window.protocol("WM_DELETE_WINDOW", on_closing)
window.mainloop()
