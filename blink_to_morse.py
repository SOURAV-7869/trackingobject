import cv2
import time
import math
import pyttsx3
import mediapipe as mp

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

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Initialize variables
blink_sequence = ""
current_word = ""
accumulated_text = ""
EAR_THRESHOLD = 0.21
BLINK_THRESHOLD = 0.3
last_blink_time = time.time()
start_time = time.time()

# Initialize webcam
cap = cv2.VideoCapture(0)

def get_pixel_coordinates(landmark, image_width, image_height):
    """Convert normalized landmark coordinates to pixel coordinates."""
    return int(landmark.x * image_width), int(landmark.y * image_height)

def calculate_ear(eye_landmarks, image_width, image_height):
    """Calculate Eye Aspect Ratio (EAR) using pixel coordinates."""
    try:
        # Convert landmarks to pixel coordinates
        landmarks = [get_pixel_coordinates(lm, image_width, image_height) for lm in eye_landmarks]

        # Calculate the distances between specific points for EAR
        vertical_1 = math.dist(landmarks[1], landmarks[5])
        vertical_2 = math.dist(landmarks[2], landmarks[4])
        horizontal = math.dist(landmarks[0], landmarks[3])

        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except (IndexError, TypeError) as e:
        print("Error calculating EAR:", e)
        return None  # Return None if there’s an issue

def morse_to_text(morse_code):
    """Convert Morse code sequence to text."""
    return morse_code_dict.get(morse_code, "")

def speak_text(text):
    """Use TTS to speak out the given text."""
    engine.say(text)
    engine.runAndWait()

def save_to_file(text):
    """Save accumulated text to a file."""
    with open("accumulated_text.txt", "w") as file:
        file.write(text)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width = frame.shape[:2]

    # Process the frame using MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Draw facial landmarks for feedback
            mp.solutions.drawing_utils.draw_landmarks(
                frame, landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )

            # Extract left and right eye landmarks
            left_eye = [
                landmarks.landmark[33], landmarks.landmark[133], landmarks.landmark[160],
                landmarks.landmark[158], landmarks.landmark[154], landmarks.landmark[144]
            ]
            right_eye = [
                landmarks.landmark[362], landmarks.landmark[263], landmarks.landmark[249],
                landmarks.landmark[253], landmarks.landmark[466], landmarks.landmark[249]
            ]

            # Calculate EAR for both eyes with error handling
            left_ear = calculate_ear(left_eye, image_width, image_height)
            right_ear = calculate_ear(right_eye, image_width, image_height)

            if left_ear is not None and right_ear is not None:
                ear = (left_ear + right_ear) / 2

                # Display EAR on the frame
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Detect blinks based on EAR threshold
                if ear < EAR_THRESHOLD:
                    blink_duration = time.time() - last_blink_time
                    if blink_duration < BLINK_THRESHOLD:
                        blink_sequence += "."
                    else:
                        blink_sequence += "-"
                    last_blink_time = time.time()

    # Translate blink sequence to Morse code and accumulate text
    if blink_sequence and (time.time() - last_blink_time) > BLINK_THRESHOLD:
        letter = morse_to_text(blink_sequence.strip())
        if letter:
            current_word += letter
        blink_sequence = ""

    # Display text on the frame
    cv2.putText(frame, f"Current Word: {current_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Blink Sequence: {blink_sequence}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Accumulated Text: {accumulated_text}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Add word to accumulated text and reset after some time
    if current_word and (time.time() - start_time) > 3:
        accumulated_text += current_word + " "
        speak_text(current_word)  # Speak the word
        current_word = ""  # Reset current word
        start_time = time.time()

    # Show the frame
    cv2.imshow("Blink to Morse Code Translator", frame)

    # Keybindings for exiting and saving
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit the program
        break
    elif key == ord('s'):  # Save accumulated text to a file
        save_to_file(accumulated_text)
        print("Accumulated text saved to accumulated_text.txt")

# Cleanup
cap.release()
cv2.destroyAllWindows()
