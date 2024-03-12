from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow import keras
import base64

app = Flask(__name__)

# Load the saved violence detection model
loaded_model = keras.models.load_model('violence_detection_model.h5')

threshold_violence = 0.7
                                                
# Load YOLOv3 model and classes
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    yolo_classes = [line.strip() for line in f.readlines()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_violence():
    frame_data = request.get_json()
    frame_base64 = frame_data['frame']
    
    # Convert the base64 frame to a NumPy array
    frame_bytes = base64.b64decode(frame_base64.split(',')[1])
    frame_np = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
    
    # Preprocess the frame for violence detection
    frame_violence = preprocess_frame(frame)  # Implement the preprocess_frame function
    
    # Make predictions using the loaded violence detection model
    predictions_violence = loaded_model.predict(np.expand_dims(frame_violence, axis=0))
    
    # Determine the violence detection result
    if predictions_violence[0][0] > threshold_violence:
        violence_detection = "Non-violent"
    else:
        violence_detection = "Violent"
    
    # Detect objects using YOLOv3
    class_names, detection_results = detect_objects_yolo(frame)
    
    return jsonify({'violence_detection': violence_detection, 'object_detection': {'class_names': class_names, 'detection_results': detection_results}})

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

if __name__ == '__main__':
    app.run(debug=True)
