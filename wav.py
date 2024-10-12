import os
import librosa
import numpy as np
import soundfile as sf
from tensorflow.keras.models import load_model

# Load the emotion detection model
model = load_model('emotion_detection_model.h5')

# Directory where audio files are stored
audio_dir = '/opt/lampp/htdocs/Speech_Emotion_Recognition/inputs/'

# Function to extract features from the audio
def extract_features(file_name):
    # Load audio file
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    
    # Extract MFCC (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)  # Take the mean
    return mfccs_processed

# Loop through all audio files in the input directory
for subdir, _, files in os.walk(audio_dir):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(subdir, file)

            # Log the name of the file (this is the comment part)
            print(f"\n# Processing file: {file}")

            # Extract features from the audio
            features = extract_features(file_path)

            # Reshape for model input
            features_reshaped = features.reshape(1, -1)

            # Predict the emotion using the model
            prediction = model.predict(features_reshaped)
            predicted_emotion = np.argmax(prediction)

            # Map prediction to emotion label (assuming a list of labels)
            emotion_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
            print(f"Predicted Emotion: {emotion_labels[predicted_emotion]}")
