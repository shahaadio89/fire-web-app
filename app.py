from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import os

app = Flask(__name__)

fire_detected = False


# =========================
# 🔥 كشف النار المحسن
# =========================
def detect_fire(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # نطاقات لون النار (أدق)
    lower1 = np.array([0, 120, 150])
    upper1 = np.array([15, 255, 255])

    lower2 = np.array([15, 120, 150])
    upper2 = np.array([35, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = mask1 + mask2

    # تنظيف الضوضاء
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    fire_pixels = cv2.countNonZero(mask)

    total_pixels = frame.shape[0] * frame.shape[1]
    fire_ratio = fire_pixels / total_pixels

    return fire_ratio > 0.02


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