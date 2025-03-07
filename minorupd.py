import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])  # Vertical distance
    B = dist.euclidean(eye[2], eye[4])  # Vertical distance
    C = dist.euclidean(eye[0], eye[3])  # Horizontal distance
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[1], mouth[7])  # Vertical distance
    B = dist.euclidean(mouth[2], mouth[6])  # Vertical distance
    C = dist.euclidean(mouth[0], mouth[4])  # Horizontal distance
    mar = (A + B) / (2.0 * C)
    return mar

# Function to calculate head pose (pitch, yaw, roll)
def head_pose_estimation(landmarks_points):
    # 3D model points for head pose estimation
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left Mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    # 2D image points from landmarks
    image_points = np.array([
        landmarks_points[30],  # Nose tip
        landmarks_points[8],   # Chin
        landmarks_points[36],  # Right eye left corner
        landmarks_points[45],  # Left eye right corner
        landmarks_points[48],  # Left mouth corner
        landmarks_points[54]   # Right mouth corner
    ], dtype="double")

    # Camera matrix and distortion coefficients
    size = frame.shape #shape = (height, width, channels)
    focal_length = size[1] #size[1] is the width of the frame
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    # Solve for head pose
    (_, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    # Calculate pitch, yaw, and roll angles from the rotation matrix
    pitch = np.degrees(np.arcsin(rotation_matrix[2, 1]))  # Pitch angle
    yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))  # Yaw angle
    roll = np.degrees(np.arctan2(rotation_matrix[2, 0], rotation_matrix[2, 2]))  # Roll angle

    return pitch, yaw, roll

# Thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for drowsiness
MAR_THRESHOLD = 1   # Mouth Aspect Ratio threshold for yawning
PITCH_THRESHOLD = 25  # Head tilt threshold (in degrees)
CONSECUTIVE_FRAMES = 20  # Number of frames for eye closure detection
YAWN_CONSECUTIVE_FRAMES = 10  # Number of frames for yawning detection

# Counters
frame_counter = 0  # Counter for eye closure
yawn_counter = 0   # Counter for yawning

# Initialize Dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indices for left and right eyes and mouth
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))
MOUTH = list(range(48, 68))

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)]

        # Extract left eye, right eye, and mouth landmarks
        left_eye = [landmarks_points[i] for i in LEFT_EYE]
        right_eye = [landmarks_points[i] for i in RIGHT_EYE]
        mouth = [landmarks_points[i] for i in MOUTH]

        # Calculate EAR and MAR
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        avg_ear = (left_ear + right_ear) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # Calculate head pose
        pitch, yaw, roll = head_pose_estimation(landmarks_points)

        # Check for drowsiness based on EAR (Eye Closure)
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!!! - EYES CLOSED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            frame_counter = 0  # Reset counter if eyes are open

        # Check for yawning
        C = dist.euclidean(mouth[0], mouth[4])  # Horizontal distance
        if mar > MAR_THRESHOLD and C > 20:  # Check if mouth is wide enough
            yawn_counter += 1
            if yawn_counter >= YAWN_CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!!! - YAWNING DETECTED", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            yawn_counter = 0  # Reset counter if mouth is not wide open

        # Check for neck tilting (forward or backward)
        if pitch > PITCH_THRESHOLD:  # Backward tilt
            cv2.putText(frame, "DROWSINESS ALERT!!! - HEAD TILT BACKWARD", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif pitch < -PITCH_THRESHOLD:  # Forward tilt
            cv2.putText(frame, "DROWSINESS ALERT!!! - HEAD TILT FORWARD", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw eye and mouth landmarks
        for point in left_eye + right_eye + mouth:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Drowsiness Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()