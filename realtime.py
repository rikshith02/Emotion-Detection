# import cv2
# from keras.models import model_from_json
# import numpy as np

# # Load model architecture from JSON file
# json_file = open(r"C:\Users\shiva sai\Desktop\Emotion Detection\emotiondetector3.json", "r")
# model_json = json_file.read()
# json_file.close()
# model = model_from_json(model_json)

# # Load model weights
# model.load_weights(r"C:\Users\shiva sai\Desktop\Emotion Detection\emotiondetector3.h5")

# # Load Haar Cascade for face detection
# haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(haar_file)

# # Function to extract features from image
# def extract_features(image):
#     feature = np.array(image)
#     feature = feature.reshape(1, 48, 48, 1)
#     return feature / 255.0

# # Dictionary for labels
# labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# # Access webcam
# webcam = cv2.VideoCapture(0)

# # Real-time facial expression detection loop
# while True:
#     ret, im = webcam.read()
#     gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
#     try:
#         for (p, q, r, s) in faces:
#             # Extract face region, preprocess, and predict
#             face_image = gray[q:q+s, p:p+r]
#             cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
#             face_image_resized = cv2.resize(face_image, (48, 48))
#             img = extract_features(face_image_resized)
#             pred = model.predict(img)
#             prediction_label = labels[pred.argmax()]
#             cv2.putText(im, '%s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
#         # Display output frame
#         cv2.imshow("Output", im)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 27:  # Exit on ESC
#             break
#     except cv2.error:
#         pass

# # Release webcam and close OpenCV windows
# webcam.release()
# cv2.destroyAllWindows()



import cv2
from keras.models import model_from_json
import numpy as np
import os

# Get the directory of the current script
project_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to your model files using relative paths
json_file_path = os.path.join(project_dir, 'emotiondetector3.json')
weights_file_path = os.path.join(project_dir, 'emotiondetector3.h5')

# Load model architecture from JSON file
with open(json_file_path, 'r') as json_file:
    model_json = json_file.read()

model = model_from_json(model_json)

# Load model weights
model.load_weights(weights_file_path)

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Dictionary for labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Access webcam
webcam = cv2.VideoCapture(0)

# Real-time facial expression detection loop
while True:
    ret, im = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    
    try:
        for (p, q, r, s) in faces:
            # Extract face region, preprocess, and predict
            face_image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            face_image_resized = cv2.resize(face_image, (48, 48))
            img = extract_features(face_image_resized)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '%s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
        
        # Display output frame
        cv2.imshow("Output", im)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Exit on ESC
            break
    except cv2.error:
        pass

# Release webcam and close OpenCV windows
webcam.release()
cv2.destroyAllWindows()
