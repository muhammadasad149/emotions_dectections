import cv2  # OpenCV for computer vision tasks
from keras.models import load_model  # Load pre-trained models using Keras
from mtcnn.mtcnn import MTCNN  # Multi-Task Cascaded Convolutional Networks for face detection
import numpy as np  # NumPy for numerical operations
import os  # Operating System module for file path handling

# Set the file path for the emotion recognition model
model_path = os.path.join(os.path.dirname(__file__), "facial_emotion_model.h5")
emotion_model = load_model(model_path)

# Load MTCNN for face detection
mtcnn = MTCNN()

# Function to predict emotions and draw bounding boxes
def predict_emotion_and_draw_box(image):
    # Convert the image to grayscale for emotion recognition
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use MTCNN for face detection
    faces = mtcnn.detect_faces(image)

    # Iterate through detected faces
    for face in faces:
        x, y, w, h = face['box']
        face_roi = gray_image[y:y + h, x:x + w]

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

        # Draw bounding box and emotion label on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Process images
image_path = 'WhatsApp Image 2024-01-02 at 6.28.18 PM.jpeg'
input_image = cv2.imread(image_path)
output_image = predict_emotion_and_draw_box(input_image)

# Display the result
cv2.imshow('Emotion Recognition', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()




