from flask import Flask, jsonify, request
import subprocess
import threading
import time
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
monitoring_process = None
drowsiness_detected = False
yawn_detected = False
forward_tilt_detected = False
backward_tilt_detected = False

last_drowsiness_time = 0
last_yawn_time = 0
last_forward_tilt_time = 0
last_backward_tilt_time = 0

should_stop = False

@app.route('/start', methods=['POST'])
def start_monitoring():
    global monitoring_process, drowsiness_detected, yawn_detected, forward_tilt_detected, backward_tilt_detected
    global last_drowsiness_time, last_yawn_time, last_forward_tilt_time, last_backward_tilt_time
    global should_stop

    if monitoring_process is not None:
        return jsonify({"message": "Monitoring is already active!"}), 400
    
    # Reset detection states
    drowsiness_detected = False
    yawn_detected = False
    forward_tilt_detected = False
    backward_tilt_detected = False

    last_drowsiness_time = 0
    last_yawn_time = 0
    last_forward_tilt_time = 0
    last_backward_tilt_time = 0

    should_stop = False
    
    # Start the face detection in a separate process
    monitoring_process = subprocess.Popen(["python3", "minor.py"])
    return jsonify({
        "message": "Monitoring started successfully!",
        "status": "active"
    }), 200

@app.route('/stop', methods=['POST'])
def stop_monitoring():
    global monitoring_process, should_stop
    if monitoring_process is None:
        return jsonify({"message": "No active monitoring process!"}), 400
    
    # Signal the process to stop
    should_stop = True
    
    # Stop the monitoring process
    monitoring_process.terminate()
    monitoring_process = None

    return jsonify({
        "message": "Monitoring stopped successfully!",
        "status": "inactive"
    }), 200

@app.route('/status', methods=['GET'])
def status():
    global drowsiness_detected, yawn_detected, forward_tilt_detected, backward_tilt_detected
    global last_drowsiness_time, last_yawn_time, last_forward_tilt_time, last_backward_tilt_time
    global monitoring_process

    current_time = time.time()

    drowsiness_active = (current_time - last_drowsiness_time) < 2 if drowsiness_detected else False
    yawn_active = (current_time - last_yawn_time) < 2 if yawn_detected else False
    forward_tilt_active = (current_time - last_forward_tilt_time) < 2 if forward_tilt_detected else False
    backward_tilt_active = (current_time - last_backward_tilt_time) < 2 if backward_tilt_detected else False

    return jsonify({
        "drowsiness": drowsiness_active,
        "yawn": yawn_active,
        "forward_tilt": forward_tilt_active,
        "backward_tilt": backward_tilt_active,
        "status": "active" if monitoring_process else "inactive",
        "timestamp": current_time
    }), 200

@app.route('/detection', methods=['POST'])
def detection_event():
    global drowsiness_detected, yawn_detected, forward_tilt_detected, backward_tilt_detected
    global last_drowsiness_time, last_yawn_time, last_forward_tilt_time, last_backward_tilt_time

    detection_type = request.json.get('type', '')

    if detection_type == 'drowsiness':
        drowsiness_detected = True
        last_drowsiness_time = time.time()
    elif detection_type == 'yawn':
        yawn_detected = True
        last_yawn_time = time.time()
    elif detection_type == 'forward_tilt':
        forward_tilt_detected = True
        last_forward_tilt_time = time.time()
    elif detection_type == 'backward_tilt':
        backward_tilt_detected = True
        last_backward_tilt_time = time.time()
    
    return jsonify({"message": f"{detection_type} detection recorded"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')
