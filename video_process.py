import imutils
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import csv


class VideoCamera:
    def __init__(self, video=None):
        self.input_file = str(video)
        self.cap = cv2.VideoCapture(video)

        detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
        emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'

        self.fd = cv2.CascadeClassifier(detection_model_path)
        self.ec = load_model(emotion_model_path, compile=False)
        self.emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised",
                         "neutral"]

        self.file = open("output/" + self.input_file.split("/")[2].split('.')[0] + '.csv', 'w')
        self.csv_file = csv.writer(self.file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csv_file.writerow(['Predicted Emotion', 'Probability'])

    def __del__(self):
        print("Destroyed")

    def get_frame(self):
        r, img = self.cap.read()

        if r is False:
            self.cap.release()
            cv2.destroyAllWindows()
            return None

        img = imutils.resize(img, width=1000)

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.fd.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                           key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
            # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
            roi = gray_img[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = self.ec.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = self.emotions[preds.argmax()]

            cv2.putText(img, label, (fX, fY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(img, (fX, fY), (fX + fW, fY + fH),
                          (0, 0, 255), 2)

            self.csv_file.writerow([label, emotion_probability])

        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()
