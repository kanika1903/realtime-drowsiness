import cv2
import dlib
from scipy.spatial import distance
import numpy as np
import requests
import time
import os

# Load pre-trained models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Constants
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness
MAR_THRESHOLD = 0.75  # Mouth Aspect Ratio threshold for yawning
PITCH_THRESHOLD = 10  # Pitch angle threshold for head tilt
CONSECUTIVE_FRAMES = 20  # Number of consecutive frames for detection
ALERT_URL = "http://localhost:5001/detection"
STATUS_CHECK_INTERVAL = 1  # Seconds between status checks

# Functions
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[0], mouth[6])
    B = distance.euclidean(mouth[2], mouth[10])
    C = distance.euclidean(mouth[4], mouth[8])
    return (B + C) / (2.0 * A)

def head_pose_estimation(frame, landmarks_points):
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    image_points = np.array([
        landmarks_points[30],  # Nose tip
        landmarks_points[8],   # Chin
        landmarks_points[36],  # Right eye left corner
        landmarks_points[45],  # Left eye right corner
        landmarks_points[48],  # Left mouth corner
        landmarks_points[54]   # Right mouth corner
    ], dtype="double")

    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))

    (_, rotation_vector, _) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    pitch = np.degrees(np.arcsin(rotation_matrix[2, 1]))
    return pitch

def check_should_stop():
    try:
        response = requests.get("http://localhost:5001/status")
        if response.status_code == 200:
            data = response.json()
            return data.get('status') != 'active'
    except:
        return True
    return False

def start_face_detection():
    cap = cv2.VideoCapture(0)
    drowsy_counter = 0
    yawn_counter = 0
    forward_tilt_counter = 0
    backward_tilt_counter = 0
    last_drowsy_alert_time = 0
    last_yawn_alert_time = 0
    last_forward_tilt_alert_time = 0
    last_backward_tilt_alert_time = 0
    
    while True:
        if check_should_stop():
            break
        
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)
            landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

            # Get eye landmarks
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
            # Get mouth landmarks
            mouth = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            
            # Calculate EAR and MAR
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = mouth_aspect_ratio(mouth)
            
            # Calculate pitch (head tilt)
            pitch = head_pose_estimation(frame, landmarks_points)

            # Check for drowsiness
            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_drowsy_alert_time > 2:
                        try:
                            requests.post(ALERT_URL, json={'type': 'drowsiness'})
                            last_drowsy_alert_time = current_time
                        except Exception as e:
                            print(f"Error sending drowsiness alert: {e}")
                        drowsy_counter = 0
            else:
                drowsy_counter = 0

            # Check for yawning
            if mar > MAR_THRESHOLD:
                yawn_counter += 1
                if yawn_counter >= CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_yawn_alert_time > 2:
                        try:
                            requests.post(ALERT_URL, json={'type': 'yawn'})
                            last_yawn_alert_time = current_time
                        except Exception as e:
                            print(f"Error sending yawn alert: {e}")
                        yawn_counter = 0
            else:
                yawn_counter = 0

            # Check for forward neck tilt
            if pitch > PITCH_THRESHOLD:
                forward_tilt_counter += 0.75
                if forward_tilt_counter >= CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_forward_tilt_alert_time > 2:
                        try:
                            requests.post(ALERT_URL, json={'type': 'forward_tilt'})
                            last_forward_tilt_alert_time = current_time
                        except Exception as e:
                            print(f"Error sending forward tilt alert: {e}")
                        forward_tilt_counter = 0
            else:
                forward_tilt_counter = 0

            # Check for backward neck tilt
            if pitch < -PITCH_THRESHOLD:
                backward_tilt_counter += 1
                if backward_tilt_counter >= CONSECUTIVE_FRAMES:
                    current_time = time.time()
                    if current_time - last_backward_tilt_alert_time > 2:
                        try:
                            requests.post(ALERT_URL, json={'type': 'backward_tilt'})
                            last_backward_tilt_alert_time = current_time
                        except Exception as e:
                            print(f"Error sending backward tilt alert: {e}")
                        backward_tilt_counter = 0
            else:
                backward_tilt_counter = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_face_detection()
