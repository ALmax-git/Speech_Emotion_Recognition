# trainer.py - This script trains a speech emotion recognition model using the dataset
# located in the 'inputs' directory and saves the trained model to a file.

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Emotion label mapping (update based on your dataset)
emotion_labels = {
    '01': 'angry',
    '02': 'disgust',
    '03': 'fear',
    '04': 'happy',
    '05': 'sad',
    '06': 'surprise',
    '07': 'neutral'
}

# Path to the dataset
input_directory = '/opt/lampp/htdocs/Speech_Emotion_Recognition/inputs'

# Initialize lists to hold features and labels
X = []
y = []

# Function to extract features from audio files
def extract_features(file_path):
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        
        # Extract various audio features
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Combine the features
        combined_features = np.concatenate((mfcc, delta_mfcc, delta2_mfcc), axis=0)
        return np.mean(combined_features, axis=1)  # Return the mean of the features
    
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# Function to extract emotion from the filename
def extract_emotion_from_filename(file_name):
    emotion_code = file_name.split("-")[2]
    emotion = emotion_labels.get(emotion_code, 'unknown')
    print(f"File: {file_name}, Emotion: {emotion}")  # Debugging line
    return emotion

# Loop through the dataset directory to extract features and labels
for folder in os.listdir(input_directory):
    folder_path = os.path.join(input_directory, folder)
    
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(folder_path, file_name)
                emotion = extract_emotion_from_filename(file_name)
                features = extract_features(file_path)
                
                if features is not None:  # Only append if features were successfully extracted
                    X.append(features)
                    y.append(emotion)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Print debug info about dataset
print("Feature matrix shape:", X.shape)
print("Labels:", np.unique(y))  # Debugging line to show all unique emotions

# Encode the labels as integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Print the encoded labels (for debugging)
print("Encoded labels:", y_encoded)

# Check if y_encoded is empty before converting to categorical
if len(y_encoded) > 0:
    y_categorical = to_categorical(y_encoded)
else:
    print("No valid emotion labels found. Exiting.")
    exit()

# Create the model
model = Sequential()
model.add(LSTM(64, input_shape=(X.shape[1], 1), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Reshape X to have 3 dimensions (samples, time steps, features)
X = np.expand_dims(X, axis=-1)

# Train the model
model.fit(X, y_categorical, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

# Save the trained model
model.save('emotion_detection_model.h5')
print("Model saved as 'emotion_detection_model.h5'")
