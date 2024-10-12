# 🎤 Speech Emotion Recognition (SER) System

This project focuses on recognizing human emotions from speech recordings. It leverages **deep learning** techniques and audio feature extraction to classify emotions such as **happy**, **sad**, **angry**, and more from `.wav` files. The system is designed to process a wide range of speech data and can be extended or adapted for various applications, including virtual assistants, mental health monitoring, and emotion-aware AI systems.

## 🎯 Project Overview

This project consists of the following key components:
- **Audio Preprocessing**: Converts audio data into machine-readable formats (MFCC, Chroma, Mel features).
- **Emotion Classification**: Uses a trained neural network to predict emotions from speech input.
- **Emotion Dataset**: Pre-processed `.wav` files organized by speaker and emotion labels.

## 🚀 Features

- **Multiclass Emotion Detection**: Recognizes emotions like *happy*, *sad*, *neutral*, *angry*, *fearful*, *disgust*, *surprised*.
- **Audio File Handling**: Supports `.wav` format, with the option to expand for other formats.
- **Deep Learning**: Utilizes **Keras** with **TensorFlow** backend for training and predictions.
- **Modular Architecture**: Easily customizable for different datasets or classification tasks.
- **Real-Time Performance** (planned): Expand to real-time emotion detection from live audio streams.

## 🛠️ Installation

Before you begin, make sure you have **Python 3.x** installed on your system.

### 1. Clone the Repository
'''bash
git clone https://github.com/ALmax-git/Speech_Emotion_Recognition.git
cd Speech_Emotion_Recognition

### Install Dependencies
- pip install -r requirements.txt

### Dataset Setup

Place your .wav audio files into the inputs/ directory. Ensure that the files are properly labeled for emotion training (follow naming conventions explained below).

## 📂 Project Structure

Speech_Emotion_Recognition/
│
├── inputs/                         # Audio dataset directory
│   ├── Actor_01/                   # Each actor's audio recordings
│   ├── Actor_02/
│   └── ...                        
│
├── trainer.py                      # Model training script
├── main.py          # Feature extraction and dataset preparation
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
└── model/                          # Saved model files (after training)


## 🧠 Model Training

The system extracts features from the speech data using MFCCs, Chroma, and Mel Spectrograms to represent the audio signal. The features are then fed into a neural network built with Keras.

- Step 1: Preprocess Audio Data
> Run the main.py script to extract features from the audio files:
- -  python3 main.py

- Step 2: Train the Model
> Once the features are extracted, train the model with:
- - python3 trainer.py

## 📝 Emotion Labels and File Naming Convention

> Each audio file should be named according to a standardized format, where the numbers represent different attributes of the recording, including the emotion:

Example:
03-01-08-02-01-02-06.wav

**Emotion codes:**

-    01 = neutral
-    02 = calm
-    03 = happy
-    04 = sad
-    05 = angry
-    06 = fearful
-    07 = disgust
-    08 = surprised

## ⚠️ Troubleshooting

>  No module named 'resampy': Ensure resampy is installed correctly using:
- pip install resampy
> Check that your input audio files are in .wav format. If necessary, convert your audio files to this format.

## 🏆 Contributing

We welcome contributions to this project! If you have ideas, suggestions, or find a bug, please feel free to submit an issue or pull request.
## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.

## 🗺️ Roadmap

* Emotion classification from .wav files.
* Add support for real-time emotion detection.
* bExpand dataset support and performance optimization.
* Improve visualization of results.























