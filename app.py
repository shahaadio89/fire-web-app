from flask import Flask, render_template, Response
import cv2
import os
import time
import pygame

app = Flask(__name__)

# =========================
# تشغيل الصوت (pygame)
# =========================
pygame.mixer.init()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(BASE_DIR, "alarm.mp3")

alarm_sound = pygame.mixer.Sound(alarm_path)

# =========================
# تحميل نموذج كشف الحريق
# =========================
xml_path = os.path.join(BASE_DIR, "fire_detection.xml")
fire_cascade = cv2.CascadeClassifier(xml_path)

print("Loaded:", not fire_cascade.empty())

# =========================
# تشغيل الكاميرا
# =========================
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

time.sleep(1)


# =========================
# بث الفيديو + كشف + صوت
# =========================
def generate_frames():
    while True:
        ret, frame = camera.read()

        if not ret or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fires = fire_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(25, 25)
        )

        # 🔥 تشغيل الإنذار الصوتي (بدون تكرار مزعج)
        if len(fires) > 0:
            if not pygame.mixer.get_busy():
                alarm_sound.play()

        # رسم المربعات
        for (x, y, w, h) in fires:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE!", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # تحويل الصورة للبث
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        if not ret:
            continue

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
# تشغيل السيرفر
# =========================
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)