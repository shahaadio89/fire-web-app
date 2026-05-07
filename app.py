from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import os

app = Flask(__name__)

fire_detected = False


# =========================
# كشف النار باللون (HSV)
# =========================
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # نطاق ألوان النار (أحمر + برتقالي + أصفر)
    lower = np.array([0, 120, 150])
    upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)

    fire_pixels = cv2.countNonZero(mask)

    return fire_pixels > 3000


# =========================
# الصفحة الرئيسية
# =========================
@app.route('/')
def index():
    return render_template('index.html')


# =========================
# استقبال صورة من الجوال
# =========================
@app.route('/detect_fire', methods=['POST'])
def detect_fire_api():
    global fire_detected

    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if detect_fire(frame):
        fire_detected = True
    else:
        fire_detected = False

    return jsonify({"fire": fire_detected})


# =========================
# حالة الحريق
# =========================
@app.route('/status')
def status():
    return jsonify({"fire": fire_detected})


# =========================
# تشغيل السيرفر
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)