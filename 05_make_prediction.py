# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable AVX instructions
import cv2
import sqlite3
import numpy as np
from keras.models import load_model

# Load pre-trained face recognition model
faceRecognizer = cv2.face.LBPHFaceRecognizer_create()
faceRecognizer.read("models/trained_lbph_face_recognizer_model.yml")

# Load Haarcascade for face detection
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# Load the gesture recognition model
gesture_model = load_model("converted_keras/keras_Model.h5", compile=False)
gesture_class_names = [line.strip() for line in open("converted_keras/labels.txt", "r").readlines()]

# Common settings for display
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.6
fontColor = (255, 255, 255)
fontWeight = 2

# Open a connection to the first webcam
camera = cv2.VideoCapture(0)

# Start looping
while True:
    # Capture frame-by-frame
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # For each face found
    for (x, y, w, h) in faces:
        # Recognize the face
        customer_uid, Confidence = faceRecognizer.predict(gray[y:y + h, x:x + w])

        # Connect to SQLite database
        try:
            conn = sqlite3.connect('customer_faces_data.db')
            c = conn.cursor()
        except sqlite3.Error as e:
            print("SQLite error:", e)

        c.execute("SELECT customer_name FROM customers WHERE customer_uid LIKE ?", (f"{customer_uid}%",))
        row = c.fetchone()
        if row:
            customer_name = row[0].split(" ")[0]
        else:
            customer_name = "Unknown"

        if 45 < Confidence < 85:
            # Create rectangle around the face
            cv2.rectangle(frame, (x - 20, y - 20), (x + w + 20, y + h + 20), (100, 180, 0), 2)

            # Display name tag and confidence score
            cv2.rectangle(frame, (x - 20, y - 50), (x + w + 20, y - 20), (100, 180, 0), -1)
            text = f"{customer_name}, Confidence: {np.round(Confidence, 2)}%"
            cv2.putText(frame, text, (x, y - 25), fontFace, fontScale, fontColor, fontWeight)

        conn.close()

    # Resize the frame for gesture recognition
    resized_image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
    resized_image = np.asarray(resized_image, dtype=np.float32).reshape(1, 224, 224, 3)
    resized_image = (resized_image / 127.5) - 1

    # Predict the gesture
    prediction = gesture_model.predict(resized_image)
    index = np.argmax(prediction)
    class_name = gesture_class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score on the frame
    text = f"{class_name}: {np.round(confidence_score * 100, 2)}%"
    cv2.putText(frame, text, (10, 30), fontFace, fontScale, fontColor, fontWeight)

    # Show the image in a window
    cv2.imshow("Webcam Image", frame)

    # Listen to the keyboard for presses
    if cv2.waitKey(1) == 27:  # 27 is the ASCII for the esc key on your keyboard
        break

# Release resources
camera.release()
cv2.destroyAllWindows()
