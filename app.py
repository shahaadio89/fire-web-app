from flask import Flask, render_template, Response, jsonify
import cv2
import os
import time

app = Flask(__name__)

# =========================
# تحميل نموذج الحريق
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(BASE_DIR, "fire_detection.xml")

fire_cascade = cv2.CascadeClassifier(xml_path)

print("Loaded:", not fire_cascade.empty())

# =========================
# تشغيل الكاميرا
# =========================
camera = cv2.VideoCapture(0)

time.sleep(1)

# متغير حالة الحريق
fire_detected = False


# =========================
# بث الفيديو
# =========================
def generate_frames():
    global fire_detected

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fires = fire_cascade.detectMultiScale(gray, 1.1, 5)

        if len(fires) > 0:
            fire_detected = True
        else:
            fire_detected = False

        for (x, y, w, h) in fires:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# =========================
# الصفحة الرئيسية
# =========================
@app.route('/')
def index():
    return render_template('index.html')


# =========================
# بث الفيديو
# =========================
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# =========================
# API لمعرفة حالة الحريق
# =========================
@app.route('/fire_status')
def fire_status():
    return jsonify({"fire": fire_detected})


# =========================
# تشغيل السيرفر
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)