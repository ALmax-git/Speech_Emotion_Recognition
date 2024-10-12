import os
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import OneHotEncoder
import tkinter as tk
from tkinter import filedialog, messagebox

# Function to extract MFCC from audio files
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Load the dataset
def load_data():
    paths = []
    labels = []

    emotion_dict = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fear',
        '07': 'disgust',
        '08': 'surprised'
    }

    for dirname, _, filenames in os.walk('ravdess_data/'):
        for filename in filenames:
            if filename.endswith('.wav'):
                paths.append(os.path.join(dirname, filename))
                emotion_code = filename.split('-')[2]
                emotion = emotion_dict.get(emotion_code, 'unknown')
                labels.append(emotion)

    print(f'Dataset Loaded: {len(paths)} samples')

    # Create a dataframe
    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels

    return df


# Preprocess data (extract features and labels)
def preprocess_data(df):
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = np.array([x for x in X_mfcc])
    X = np.expand_dims(X, -1)

    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']])
    y = y.toarray()

    return X, y

# Build the LSTM model
def build_model():
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')  # Assuming 7 emotion labels
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Function to predict emotion
def predict_emotion(input_path, model):
    data, sr = librosa.load(input_path, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr, n_mfcc=40).T, axis=0)
    mfcc = np.expand_dims(mfcc, axis=0)
    mfcc = np.expand_dims(mfcc, axis=-1)

    predicted_probabilities = model.predict(mfcc)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
    predicted_emotion = emotion_labels[np.argmax(predicted_probabilities)]

    return predicted_emotion

# Train and save the model
def train_model():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=64)

    model.save('emotion_detection_model.h5')
    print("Model trained and saved.")

    return model, history

# Plot accuracy and loss
def plot_history(history):
    epochs = list(range(50))
    
    # Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.plot(epochs, loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Load trained model from file
def load_trained_model():
    model = tf.keras.models.load_model('emotion_detection_model.h5')
    return model

# GUI implementation using Tkinter
def create_gui():
    model = load_trained_model()  # Load the pre-trained model

    # Function to select and predict emotion from an audio file
    def recognize_emotion():
        file_path = filedialog.askopenfilename(title="Select Audio File", filetypes=(("Audio files", "*.wav"), ("All files", "*.*")))
        if file_path:
            selected_audio_label.config(text=f"Selected Audio: {file_path}")
            selected_audio_path.set(file_path)

    # Function to predict emotion and display it
    def predict_and_display_emotion():
        file_path = selected_audio_path.get()
        if file_path:
            try:
                predicted_emotion = predict_emotion(file_path, model)
                result_label.config(text=f"Predicted Emotion: {predicted_emotion}")
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred: {str(e)}")

    # Main application window
    app = tk.Tk()
    app.title("Speech Emotion Recognition System")

    # Welcome label
    welcome_label = tk.Label(app, text="Welcome to the Speech Emotion Recognition System", font=("Helvetica", 16))
    welcome_label.pack(pady=10)

    # Create GUI components
    select_button = tk.Button(app, text="Select Audio", command=recognize_emotion)
    selected_audio_label = tk.Label(app, text="Selected Audio: ")
    selected_audio_path = tk.StringVar()
    result_label = tk.Label(app, text="Predicted Emotion: ")

    # Predict button
    predict_button = tk.Button(app, text="Predict Emotion", command=predict_and_display_emotion)

    # Place components in the window
    select_button.pack()
    selected_audio_label.pack()
    predict_button.pack()
    result_label.pack()

    # Start the GUI main loop
    app.mainloop()

# Train the model and create GUI
if __name__ == "__main__":
    # Uncomment the next line if you need to train the model first
    # model, history = train_model()
    # plot_history(history)  # Visualize accuracy and loss after training

    # Run the GUI application
    create_gui()
