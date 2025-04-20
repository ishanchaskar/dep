import streamlit as st
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import gdown
import tempfile
import time

# Download model from Google Drive
file_id = "1WeSZoGTIvp1qkhuAxIiDmSQVIYoWIoXb"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "asl_model.h5"
labels_path = "labels.txt"

# Download once
@st.cache_resource
def download_model():
    gdown.download(url, model_path, quiet=False)
    return model_path

model_file = download_model()
classifier = Classifier(model_file, labels_path)

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
labels = [chr(i) for i in range(65, 91)]  # A-Z

st.title("ASL Hand Sign Detection")
run = st.checkbox('Start Camera')

FRAME_WINDOW = st.image([])

cap = None
if run:
    cap = cv2.VideoCapture(0)
    while run:
        success, img = cap.read()
        if not success:
            st.write("Failed to grab frame")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        try:
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap: wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite)
                confidence = max(prediction) * 100
                predicted_label = labels[index]

                cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 2)
                cv2.putText(imgOutput, f"{predicted_label} ({confidence:.2f}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                cv2.putText(imgOutput, "No hand detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        except Exception as e:
            st.write("Error:", e)

        imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(imgOutput)

    cap.release()
