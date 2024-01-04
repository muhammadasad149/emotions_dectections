import cv2  # OpenCV for computer vision tasks
from keras.models import load_model  # Load pre-trained models using Keras
from mtcnn.mtcnn import MTCNN  # Multi-Task Cascaded Convolutional Networks for face detection
import numpy as np  # NumPy for numerical operations
import os  # Operating System module for file path handling

# Load the emotion recognition model
model_path = os.path.join(os.path.dirname(__file__), "facial_emotion_model.h5")
emotion_model = load_model(model_path)

# Load MTCNN for face detection
mtcnn = MTCNN()

# Function to predict emotions and draw bounding boxes
def predict_emotion_and_draw_box(frame):
    # Convert the frame to grayscale for emotion recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Use MTCNN for face detection
    faces = mtcnn.detect_faces(frame)

    # Iterate through detected faces
    for face in faces:
        x, y, w, h = face['box']
        face_roi = gray_frame[y:y + h, x:x + w]

        # Resize the face image to the input dimensions expected by the emotion recognition model
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape((1, 48, 48, 1))
        face_roi = face_roi / 255.0  # Normalize the pixel values

        # Predict emotion using the emotion recognition model
        emotion_probabilities = emotion_model.predict(face_roi)
        emotion_label = np.argmax(emotion_probabilities)

        # Map the emotion label to a string
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = emotion_labels[emotion_label]

        # Draw bounding box and emotion label on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Process video
video_path = 'vid3.mp4'
cap = cv2.VideoCapture(video_path)

# Loop through video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Call the function to predict emotions and draw bounding boxes
    output_frame = predict_emotion_and_draw_box(frame)

    # Display the result
    cv2.imshow('Emotion Recognition', output_frame)

    # Press 'Esc' to exit the loop and close the video window
    if cv2.waitKey(500) & 0xFF == 27:
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
