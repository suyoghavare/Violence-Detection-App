from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import cv2
import numpy as np
from tensorflow import keras
import base64
from telegram import Bot
from telegram.error import TelegramError
import asyncio
from io import BytesIO
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the new violence detection model
new_loaded_model = keras.models.load_model('modelnew.h5')

threshold_violence = 0.7

# Telegram Bot API Token
TELEGRAM_API_TOKEN = '7175199576:AAGD4C24uoPKGPProRR60Uxx9SzguZq2fks'

# Chat ID for sending messages to your Telegram account
TELEGRAM_CHAT_ID = '1351987052'

# Load YOLOv3 model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

# Counter for consecutive violence detections
global consecutive_violence_count

# Define a dictionary to store users and their passwords
USERS = {'admin': 'password'}

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS and USERS[username] == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid username or password')
    return render_template('login.html')

# Logout route
@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Registration route (accessible only by admin)
@app.route('/register', methods=['GET', 'POST'])
def register():
    if not session.get('logged_in') or session['username'] != 'admin':
        return redirect(url_for('login'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in USERS:
            error = 'Username already exists'
        else:
            USERS[username] = password
            success = 'User registered successfully'
            return render_template('register.html', success=success)
    return render_template('register.html')


# Index route (protected route)
@app.route('/')
def index():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return redirect(url_for('violence_detection'))

# Violence detection route (protected)
@app.route('/violence_detection')
def violence_detection():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('index.html')

# Detect violence route
@app.route('/detect', methods=['POST'])
# Detect violence route
@app.route('/detect', methods=['POST'])
def detect_violence():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    global consecutive_violence_count  # Declare consecutive_violence_count as global
    
    frame_data = request.get_json()
    frame_base64 = frame_data['frame']
    
    # Convert the base64 frame to a NumPy array
    frame_bytes = base64.b64decode(frame_base64.split(',')[1])
    frame_np = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    # Detect objects using YOLOv3
    class_names, detection_results = detect_objects_yolo(frame)
    
    # Check if "person" is detected
    if "person" in class_names:
        # Preprocess the frame for violence detection
        frame_violence = preprocess_frame(frame)
        
        # Make predictions using the new violence detection model
        predictions_violence = new_loaded_model.predict(np.expand_dims(frame_violence, axis=0))
        
        # Determine the violence detection result
        if predictions_violence[0][0] > threshold_violence:
            violence_detection = "violence detected"
            consecutive_violence_count += 1
            if consecutive_violence_count >= 5:
                send_telegram_message("Violence detected!", frame)
                consecutive_violence_count = 0
        else:
            violence_detection = "stable"
            consecutive_violence_count = 0  # Reset the counter if violence is not detected
        
        # Print the violence detection result
        print("Violence Detection Result:", violence_detection)
        
        return jsonify({'violence_detection': violence_detection, 'object_detection': {'class_names': class_names, 'detection_results': detection_results}})
    else:
        consecutive_violence_count = 0  # Reset the counter if no person is detected
        return jsonify({'violence_detection': 'No action detection triggered because no person detected', 'object_detection': {'class_names': class_names, 'detection_results': detection_results}})

def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))
    frame = frame / 255.0
    return frame

def detect_objects_yolo(frame):
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    class_names = []
    detection_results = []

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            class_name = yolo_classes[class_ids[i]]
            confidence = confidences[i]
            class_names.append(class_name)
            detection_results.append(f"{class_name} {confidence:.2f}")

    return class_names, detection_results

def send_telegram_message(message, frame):
    try:
        # Create a Bot instance with your API token
        bot = Bot(token=TELEGRAM_API_TOKEN)
        
        # Convert frame to JPEG
        frame_bytes = cv2.imencode('.jpg', frame)[1].tostring()
        
        # Create BytesIO object
        bio = BytesIO(frame_bytes)
        bio.name = 'image.jpg'
        
        # Send a message with the image
        asyncio.run(bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=bio, caption=message))
        print("Message sent successfully!")
    except TelegramError as e:
        print(f"Error sending message: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
