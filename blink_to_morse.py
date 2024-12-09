import cv2
import time
import math
import pyttsx3
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# Morse Code Dictionary for A-Z
morse_code_dict = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E",
    "..-.": "F", "--.": "G", "....": "H", "..": "I", ".---": "J",
    "-.-": "K", ".-..": "L", "--": "M", "-.": "N", "---": "O",
    ".--.": "P", "--.-": "Q", ".-.": "R", "...": "S", "-": "T",
    "..-": "U", "...-": "V", ".--": "W", "-..-": "X", "-.--": "Y",
    "--..": "Z"
}

# Initialize pyttsx3 for Text-to-Speech (TTS)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)
hands = mp_hands.Hands(max_num_hands=2)

# Constants
EAR_THRESHOLD = 0.21
BLINK_THRESHOLD = 0.3
PLOT_WINDOW = 50  # Number of EAR values to plot in the graph
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Variables
blink_sequence = ""
current_word = ""
accumulated_text = ""
ear_values = deque(maxlen=PLOT_WINDOW)  # For real-time plotting
last_blink_time = time.time()
start_time = time.time()

# Webcam setup
cap = cv2.VideoCapture(0)

# Plotting setup
plt.ion()
fig, ax = plt.subplots()
ear_plot, = ax.plot([], [], label="EAR", color='b')
ax.set_xlim(0, PLOT_WINDOW)
ax.set_ylim(0, 0.5)
ax.set_title("EAR Values Over Time")
ax.set_xlabel("Frames")
ax.set_ylabel("EAR")
ax.legend()

# Logging setup
log_file = open("blink_log.txt", "w")
log_file.write("Time, EAR\n")

def log_event(event, ear):
    """Log blink events with timestamp and EAR value."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_file.write(f"{timestamp}, {ear:.2f}\n")

def update_plot(ear):
    """Update real-time EAR plot."""
    ear_values.append(ear)
    ear_plot.set_ydata(ear_values)
    ear_plot.set_xdata(range(len(ear_values)))
    ax.set_xlim(0, len(ear_values))
    plt.draw()
    plt.pause(0.01)

def calculate_ear(eye_landmarks, width, height):
    """Calculate the Eye Aspect Ratio (EAR)."""
    try:
        landmarks = [
            (int(landmark.x * width), int(landmark.y * height)) for landmark in eye_landmarks
        ]
        vertical_1 = math.dist(landmarks[1], landmarks[5])
        vertical_2 = math.dist(landmarks[2], landmarks[4])
        horizontal = math.dist(landmarks[0], landmarks[3])
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except Exception as e:
        print(f"EAR calculation error: {e}")
        return None

def morse_to_text(morse_code):
    """Convert Morse code sequence to text."""
    return morse_code_dict.get(morse_code, "")

def speak_text(text):
    """Use TTS to speak out the text."""
    engine.say(text)
    engine.runAndWait()

def handle_gestures(hand_landmarks, width, height):
    """Detect specific gestures for commands."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb = (int(thumb_tip.x * width), int(thumb_tip.y * height))
    index = (int(index_tip.x * width), int(index_tip.y * height))

    # Detect pinch (thumb close to index)
    if math.dist(thumb, index) < 30:
        return "pinch"

    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not accessible.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    if face_results.multi_face_landmarks:
        for landmarks in face_results.multi_face_landmarks:
            left_eye = [
                landmarks.landmark[33], landmarks.landmark[133], landmarks.landmark[160],
                landmarks.landmark[158], landmarks.landmark[154], landmarks.landmark[144]
            ]
            right_eye = [
                landmarks.landmark[362], landmarks.landmark[263], landmarks.landmark[249],
                landmarks.landmark[253], landmarks.landmark[466], landmarks.landmark[249]
            ]

            left_ear = calculate_ear(left_eye, width, height)
            right_ear = calculate_ear(right_eye, width, height)

            if left_ear and right_ear:
                ear = (left_ear + right_ear) / 2
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), FONT, 0.6, (255, 0, 0), 2)

                update_plot(ear)
                log_event("EAR", ear)

                if ear < EAR_THRESHOLD:
                    blink_duration = time.time() - last_blink_time
                    blink_sequence += "." if blink_duration < BLINK_THRESHOLD else "-"
                    last_blink_time = time.time()

    if blink_sequence and (time.time() - last_blink_time) > BLINK_THRESHOLD:
        letter = morse_to_text(blink_sequence.strip())
        if letter:
            current_word += letter
        blink_sequence = ""

    if current_word and (time.time() - start_time) > 3:
        accumulated_text += current_word + " "
        speak_text(current_word)
        current_word = ""
        start_time = time.time()

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            gesture = handle_gestures(hand_landmarks, width, height)
            if gesture == "pinch":
                print("Gesture detected: Pinch (Clearing current word)")
                current_word = ""

    cv2.putText(frame, f"Word: {current_word}", (10, 70), FONT, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Text: {accumulated_text}", (10, 100), FONT, 0.5, (0, 255, 255), 1)

    cv2.imshow("Blink to Morse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
plt.close()
