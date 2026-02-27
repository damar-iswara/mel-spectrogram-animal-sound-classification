# mel-spectrogram-animal-sound-classification
This project aims to classify various animal sounds using digital signal processing and deep learning techniques. The core approach involves converting raw audio data into 2D visual representations called **Mel-Spectrograms**, which are then evaluated and classified using a neural network model.

kaggle-link: https://www.kaggle.com/code/damariswara/mel-spectogram-animal-sound-classification

## Project Description

Animal sounds possess unique frequency and amplitude patterns. Instead of processing the audio data in its 1D waveform state, this project transforms the audio into 2D images (Mel-Spectrograms). This visual representation allows us to leverage standard Computer Vision architectures to recognize specific sound patterns of different animals with higher accuracy.

## Libraries and Tools Used
- **Audio Processing:** `librosa`
- **Deep Learning / ML:** `TensorFlow` / `Keras`, `scikit-learn`
- **Data Manipulation:** `numpy`, `pandas`
- **Visualization:** `matplotlib`, `seaborn`

## Methodology

The pipeline for this project consists of the following steps:

1. **Data Collection & Exploratory Data Analysis (EDA):** Loading the animal sound audio files, playing audio samples, and visualizing the original waveforms to understand the data distribution.
2. **Audio Preprocessing:** Cleaning up noise, trimming silent parts, and ensuring all audio files have a uniform duration and sample rate.
3. **Feature Extraction:** - Utilizing the Fast Fourier Transform (FFT) to convert signals from the time domain to the frequency domain.
   - Mapping the frequencies to the Mel scale to mimic human auditory perception.
   - Generating and saving the **Mel-Spectrogram** images for each audio sample.
4. **Data Preparation:** Applying label encoding to the target animal classes and splitting the dataset into training, validation, and testing sets.
5. **Model Training:** Training the deep learning model using the extracted Mel-Spectrogram images as inputs.
6. **Evaluation:** Assessing the model's performance using metrics such as Accuracy, Precision, Recall, and generating a Confusion Matrix to analyze misclassifications in detail.

## Algorithm

The primary algorithm utilized in this project is a **Convolutional Neural Network (CNN)**.

Since the audio features have been converted into images (Mel-Spectrograms), a CNN is the most optimal choice. The architecture generally works through:
- **Convolutional Layers (Conv2D):** Extracting spatial features like lines, edges, and texture patterns from the frequency spectrum.
- **Pooling Layers (MaxPooling2D):** Downsampling the feature maps to reduce computational load and prevent overfitting.
- **Flatten & Dense Layers:** Flattening the 2D matrices into 1D vectors and processing them through Fully Connected Layers to output prediction probabilities.
- **Softmax Activation:** Applied at the final layer to classify the sounds into multi-class probabilities (determining the specific animal).

---
*Developed for experimental audio classification utilizing Mel-Spectrogram feature extraction.*
