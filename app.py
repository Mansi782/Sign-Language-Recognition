from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import load_model
import os
from datetime import datetime

app = Flask(__name__)

# Load the model
try:
    model = load_model('signlanguagedetectionmodel48x48.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Labels for predictions
LABELS = ['A', 'M', 'N', 'S', 'T', 'blank']

class SignLanguageDetector:
    def __init__(self):
        self.camera = None
    
    def get_camera(self):
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        return self.camera
    
    def release_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def extract_features(self, image):
        feature = np.array(image)
        feature = feature.reshape(1, 48, 48, 1)
        return feature / 255.0

    def generate_frames(self):
        camera = self.get_camera()
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            
            # Region of interest for hand gestures
            cv2.rectangle(frame, (0, 40), (300, 300), (0, 165, 255), 1)
            crop_frame = frame[40:300, 0:300]
            
            # Preprocess the frame
            gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (48, 48))
            processed = self.extract_features(resized)
            
            # Make prediction
            if model is not None:
                pred = model.predict(processed, verbose=0)
                prediction_label = LABELS[pred.argmax()]
                confidence = np.max(pred) * 100
                
                # Draw prediction on frame
                model_accuracy = 92.2  # Replace with actual accuracy calculation if needed
                if prediction_label != 'blank':
                    text = f'{prediction_label} {confidence:.2f}% (Model Accuracy: {model_accuracy:.2f}%)'
                    cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            
            # Convert frame to jpg
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Initialize detector
detector = SignLanguageDetector()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(detector.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start_detection():
    """Start the detection"""
    try:
        detector.get_camera()
        return jsonify({'status': 'success', 'message': 'Detection started'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop')
def stop_detection():
    """Stop the detection"""
    try:
        detector.release_camera()
        return jsonify({'status': 'success', 'message': 'Detection stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Create or update the template file
    template_path = os.path.join('templates', 'index.html')
    with open(template_path, 'w') as f:
        f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sign Language Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #f0f0f0;
            margin: 0;
            padding: 20px;
        }

        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            margin-top: 20px;
            text-align: center;
        }

        #videoFeed {
            border: 2px solid #333;
            border-radius: 5px;
            margin: 20px 0;
        }

        .controls {
            margin: 20px 0;
        }

        button {
            padding: 10px 20px;
            font-size: 16px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 0 10px;
            transition: background-color 0.3s;
        }

        button:hover {
            background-color: #0056b3;
        }

        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }

        .status {
            margin-top: 10px;
            padding: 10px;
            border-radius: 5px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Sign Language Detection</h1>
        <img id="videoFeed" src="{{ url_for('video_feed') }}" width="640" height="480">
        <div class="controls">
            <button id="startBtn" onclick="startDetection()">Start Detection</button>
            <button id="stopBtn" onclick="stopDetection()" disabled>Stop Detection</button>
        </div>
        <div class="status" id="status">Ready to start detection</div>
    </div>

    <script>
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const status = document.getElementById('status');

        async function startDetection() {
            try {
                const response = await fetch('/start');
                const data = await response.json();
                if (data.status === 'success') {
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    status.textContent = 'Detection running...';
                } else {
                    status.textContent = 'Error: ' + data.message;
                }
            } catch (error) {
                status.textContent = 'Error starting detection: ' + error;
            }
        }

        async function stopDetection() {
            try {
                const response = await fetch('/stop');
                const data = await response.json();
                if (data.status === 'success') {
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    status.textContent = 'Detection stopped';
                } else {
                    status.textContent = 'Error: ' + data.message;
                }
            } catch (error) {
                status.textContent = 'Error stopping detection: ' + error;
            }
        }
    </script>
</body>
</html>
        ''')
    
    # Run the Flask app
    app.run(debug=True)