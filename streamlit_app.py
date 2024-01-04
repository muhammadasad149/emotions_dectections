import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import os
import io
import tempfile

# Load the emotion recognition model
model_path = os.path.join(os.path.dirname(__file__), "facial_emotion_model.h5")
emotion_model = load_model(model_path)

# Load MTCNN for face detection
mtcnn = MTCNN()


# Function to predict emotions and draw bounding boxes for images
def process_image(uploaded_image):
    # Decode image and predict emotions
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    return predict_emotion_and_draw_box(image)

# Function to predict emotions and draw bounding boxes for videos
def process_video(uploaded_video):
    # Read video bytes and save to temporary file
    video_bytes = uploaded_video.read()
    video_np = np.frombuffer(video_bytes, dtype=np.uint8)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_np.tobytes())
    temp_file.close()

    # Open video file with cv2.VideoCapture
    cap = cv2.VideoCapture(temp_file.name)

    frames = []
    frame_index = 0  # Track the frame index

    # Process each frame in the video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predict emotion and draw bounding box
        output_frame = predict_emotion_and_draw_box(frame)
        frames.append(output_frame)

        # Display frame with a unique button for each frame
        if st.button(f'Stop ({frame_index + 1}/{len(frames)})'):
            break

        frame_index += 1

    # Cleanup: Delete the temporary file
    os.remove(temp_file.name)
    return frames

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

# Streamlit app
def main():
    st.title("Emotion Recognition App")

    # File upload options for images and videos
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    uploaded_video = st.file_uploader("Choose a video", type=["mp4"])

    # Button to open webcam
    if st.button("Open Webcam"):
        st.markdown("### Emotion Recognition via Webcam")
        video_feed = st.empty()
        stop_button = st.button('Stop Webcam')

        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture video stream")
                break

            output_frame = predict_emotion_and_draw_box(frame)
            video_feed.image(output_frame, channels="BGR", caption="Emotion Recognition Result", use_column_width=True)

        cap.release()


    if uploaded_image is not None:
        output_image = process_image(uploaded_image)
        st.image(output_image, channels="BGR", caption="Emotion Recognition Result", use_column_width=True)

    elif uploaded_video is not None:
        frames = process_video(uploaded_video)
        for output_frame in frames:
            st.image(output_frame, channels="BGR", caption="Emotion Recognition Result", use_column_width=True)

    else:
        st.warning("Please choose an image or video to process.")

if __name__ == "__main__":
    main()








