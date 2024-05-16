import torch
import numpy as np
import cv2
import os
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time
from datetime import datetime, timedelta

password = "app password"
from_email = "email address"  # must match the email used to generate the password
to_email = "where do you want the email sent to"  # receiver email

server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()
server.login(from_email, password)


import time
from datetime import datetime, timedelta

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.last_email_time = datetime.now() - timedelta(minutes=1)  # allow immediate email
        self.model = YOLO("yolov8n.pt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.recording = False
        self.out = None
        self.recording_directory = "recording" # Change this to the folder you want the recording to be saved in 

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        objects_detected = {}
        annotator = Annotator(frame, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        for box, cls in zip(boxes, clss):
            label = results[0].names[int(cls)]
            annotator.box_label(box, label=label, color=colors(int(cls), True))
            if label in objects_detected:
                objects_detected[label] += 1
            else:
                objects_detected[label] = 1
        return frame, objects_detected

    def send_email(self, objects_detected):
        if 'person' in objects_detected and objects_detected['person'] > 0:
            current_time = datetime.now()
            if current_time - self.last_email_time >= timedelta(minutes=1):
                message = MIMEMultipart()
                message['From'] = from_email
                message['To'] = to_email
                message['Subject'] = "Security Alert - Person Detected"
                body = "ALERT: Person detection summary:\n"
                body += "\n".join([f"{obj}: {count}" for obj, count in objects_detected.items()])
                message.attach(MIMEText(body, 'plain'))
                server.sendmail(from_email, to_email, message.as_string())
                self.last_email_time = current_time

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened(), "Failed to open video capture."
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        while True:
            ret, frame = cap.read()
            assert ret, "Failed to read from video capture."
            results = self.predict(frame)
            frame, objects_detected = self.plot_bboxes(results, frame)

            if objects_detected.get('person', 0) > 0:
                if not self.recording:
                    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                    video_path = os.path.join(self.recording_directory, f'recording_{timestamp}.mp4')
                    self.out = cv2.VideoWriter(video_path, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
                    self.recording = True
                self.send_email(objects_detected)

            if self.recording:
                self.out.write(frame)
                if objects_detected.get('person', 0) == 0:
                    self.out.release()
                    self.recording = False

            cv2.imshow('YOLOv8 Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()
