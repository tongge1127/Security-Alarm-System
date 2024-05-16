# Object Detection System Using YOLOv8

## Introduction
This Object Detection System leverages the YOLOv8 model to detect objects in real-time through video feeds. It is specifically tailored to detect persons within the video frame. Upon detection of a person, the system begins recording and sends an email notification. Recording ceases when no persons are detected.

## Features
- **Real-Time Detection:** Uses YOLOv8 for efficient and accurate real-time object detection.
- **Email Notifications:** Sends email alerts when a person is detected.
- **Automated Recording:** Automatically starts and stops recording based on person detection.

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- YOLOv8 model file (`yolov8n.pt`)
- SMTP library for Python (for email functionality)

## Setup
1. **Clone the Repository:**
   ```bash
   git clone https://your-repository-url.git
   cd your-repository-directory
    ```
2. **Email Configuration:**
  Configure the SMTP settings in the script to enable email functionality. You need to set the sender email, receiver email, and SMTP server details.
