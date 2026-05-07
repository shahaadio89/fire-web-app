from flask import Flask, render_template, jsonify
import cv2
import numpy as np
import os

app = Flask(__name__)

# تحميل نموذج الحريق
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "fire_detection.xml")

fire_cascade = cv2.CascadeClassifier(xml_path)

print("Loaded:", not fire_cascade.empty())

# حالة الحريق
fire_detected = False


@app.route('/')
def index():
    return render_template('index.html')


# API استقبال صورة من الجوال وتحليلها
@app.route('/detect_fire', methods=['POST'])
def detect_fire():
    global fire_detected

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    fires = fire_cascade.detectMultiScale(gray, 1.1, 5)

    if len(fires) > 0:
        fire_detected = True
    else:
        fire_detected = False

    return jsonify({"fire": fire_detected})


# API لحالة الحريق
@app.route('/status')
def status():
    return jsonify({"fire": fire_detected})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)