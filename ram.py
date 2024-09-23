from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import numpy as np
from keras.models import load_model
from playsound import playsound
import mediapipe as mp
import math
import asyncio
from threading import Thread

app = FastAPI()

model = load_model('gender_detection_model1.keras')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

templates = Jinja2Templates(directory="templates")

# Cache variables
frame_counter = 0
FRAME_SKIP = 5  # Skip gender detection every 5 frames to reduce overhead

def calculate_distance(point1, point2):
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def detect_thumb_position(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    
    palm_center_x = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].x +
                     landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].x +
                     landmarks[mp_hands.HandLandmark.PINKY_MCP].x) / 3
    palm_center_y = (landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP].y +
                     landmarks[mp_hands.HandLandmark.RING_FINGER_MCP].y +
                     landmarks[mp_hands.HandLandmark.PINKY_MCP].y) / 3
    
    thumb_to_palm_distance = math.sqrt(
        (thumb_tip.x - palm_center_x) ** 2 + 
        (thumb_tip.y - palm_center_y) ** 2)
    
    if thumb_tip.y > index_finger_mcp.y and thumb_to_palm_distance < 0.12:
        return "Tuck Thumb"
    
    return None

def is_hand_open(landmarks):
    finger_distances = []
    for finger_tip, finger_mcp in [(mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)]:
        tip = landmarks[finger_tip]
        mcp = landmarks[finger_mcp]
        distance = calculate_distance(tip, mcp)
        finger_distances.append(distance)
    
    return all(distance > 0.2 for distance in finger_distances)

async def generate_frames():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            global frame_counter
            frame_counter += 1

            # Skip gender detection on certain frames to reduce latency
            if frame_counter % FRAME_SKIP == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        if not is_hand_open(hand_landmarks.landmark):
                            for (x, y, w, h) in faces:
                                face = frame[y:y + h, x:x + w]
                                face = cv2.resize(face, (150, 150))
                                face = np.expand_dims(face, axis=0)
                                face = face / 255.0

                                prediction = model.predict(face)
                                confidence = prediction[0][0]
                                gender = "Male" if confidence > 0.5 else "Female"

                                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                                cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/video_feed')
async def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("web7.html", {"request": request})


def detect_gesture():
    global emergency_message_flag, lone_woman_flag, help_gesture_detected,male_count,female_count,surrounded

    if lone_woman_flag and help_gesture_detected:
        playsound('alone_women_gesture.mp3')
        return "Lone Women detected with help Gesture, police report to the location immediately"
    
    elif surrounded:
        playsound('female_men.mp3')
        return "One female is surrounded by 3 or more men! keep the scene in observation"
    elif lone_woman_flag:
        playsound('alone_women.mp3')
        return "Alert: Lone Female Detected! Keep her in observation."
    
    elif emergency_message_flag:
        playsound('emergency.mp3')
        return "Emergency situation!!! Needed help"
    else:
        playsound('No_unusual.mp3')
        return "No Unusual activities detected"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
