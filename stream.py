import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import gdown
from tensorflow.keras.models import load_model

# Page title
st.title("ASL Hand Gesture Recognition")

# Download model from Google Drive if not already present
file_id = "1WeSZoGTIvp1qkhuAxIiDmSQVIYoWIoXb"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "asl_model.h5"

with st.spinner("Downloading model..."):
    gdown.download(url, model_path, quiet=False)
st.success("Model downloaded successfully!")

# Initialize video capture and models
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(model_path, "labels.txt")

offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
          "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
          "U", "V", "W", "X", "Y", "Z"]

stframe = st.empty()

run = st.checkbox("Start Camera")

while run:
    success, img = cap.read()
    if not success:
        st.error("Failed to access camera.")
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
        st.warning(f"An error occurred: {e}")

    stframe.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), channels="RGB")
