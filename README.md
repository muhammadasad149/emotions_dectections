# Emotion Recognition App

This is a Streamlit-based web application for real-time emotion recognition in images and videos using deep learning models. It detects faces in the input media, predicts the emotion associated with each face, and draws bounding boxes around the faces with corresponding emotion labels.

## Features

- Emotion recognition in images and videos
- Real-time webcam-based emotion recognition
- Support for multiple emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Easy-to-use interface powered by Streamlit

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/muhammadasad149/emotions_dectections.git
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

2. Choose an image or video file for emotion recognition.
3. Optionally, open the webcam for real-time emotion recognition.

## File Structure

- `streamlit_app.py`: Main Streamlit application script.
- `facial_emotion_model.h5`: Pre-trained Keras model for facial emotion recognition.
- `requirements.txt`: List of Python dependencies.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.
