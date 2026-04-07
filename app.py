import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from math import hypot
import streamlit.components.v1 as components

st.markdown("""
<style>
/* background utama */
.stApp {
    background-color: #ff006d;
}

/* sidebar juga pink */
[data-testid="stSidebar"] {
    background-color: #00000;
}

/* teks jadi pink */
h1, h2, h3, h4 {
    color: #000000;
}

/* tombol checkbox */
.stCheckbox label {
    color: #ff4da6;
    font-weight: bold;
}

/* alert box biar lucu */
.stAlert {
    border-radius: 15px;
}

/* hilangin background gelap container */
.block-container {
    background-color: #ff5a8;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🌸 Deteksi Kantuk AI 🌸</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ff66b2;'>Stay awake bestie 😴✨</p>", unsafe_allow_html=True)
st.image("https://png.pngtree.com/png-clipart/20221005/original/pngtree-pink-flower-decoration-png-image_8668123.png", width=150)
st.markdown("🌸 🌷 🌸 🌷 🌸 🌷 🌸")

run = st.checkbox("Aktifkan Kamera")

FRAME_WINDOW = st.image([])

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.23
COUNTER = 0
EAR_CONSEC_FRAMES = 15

def eye_aspect_ratio(eye):
    A = hypot(eye[1][0] - eye[5][0], eye[1][1] - eye[5][1])
    B = hypot(eye[2][0] - eye[4][0], eye[2][1] - eye[4][1])
    C = hypot(eye[0][0] - eye[3][0], eye[0][1] - eye[3][1])
    return (A + B) / (2.0 * C) if C != 0 else 0

cap = cv2.VideoCapture(0)

status_text = st.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        mesh = results.multi_face_landmarks[0].landmark
        h, w, _ = frame.shape

        left_eye = [(int(mesh[i].x*w), int(mesh[i].y*h)) for i in LEFT_EYE]
        right_eye = [(int(mesh[i].x*w), int(mesh[i].y*h)) for i in RIGHT_EYE]

        # 🔥 GAMBAR GARIS HIJAU
        cv2.polylines(frame_rgb, [np.array(left_eye)], True, (0,255,0), 1)
        cv2.polylines(frame_rgb, [np.array(right_eye)], True, (0,255,0), 1)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < EAR_THRESHOLD:
            COUNTER += 1

            if COUNTER >= EAR_CONSEC_FRAMES:
                status_text.error("😴 Ih ngantuk banget sih! Bangun dong bestie 💖")


        else:
            COUNTER = 0
            status_text.success("😃 Masih semangat! Jangan tidur ya 🌸✨")

    FRAME_WINDOW.image(frame_rgb)

cap.release()