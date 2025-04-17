import streamlit as st
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write, read
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from pathlib import Path
import io

# Custom CSS styling
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #FFFFFF;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #6B46C1;
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #553C9A;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* File uploader styling */
    .stFileUploader>div {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 20px;
    }
    .stFileUploader>div>div>div>div {
        color: #6B46C1;
    }
    .stFileUploader>div>div>div>div>div {
        background-color: #6B46C1;
        color: white;
    }
    
    /* Headings */
    h1 {
        color: #6B46C1;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
    }
    h2 {
        color: #4A5568;
        font-size: 1.8rem;
        margin-top: 2rem;
    }
    
    /* Text */
    .stMarkdown {
        color: #4A5568;
    }
    
    /* Success message */
    .success {
        background-color: #F0F9FF;
        color: #6B46C1;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #6B46C1;
        margin-top: 1rem;
    }
    
    /* App container */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #FFFFFF;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #FFFFFF;
    }
    
    /* Streamlit default elements */
    .stAlert {
        background-color: #F0F9FF;
        border-color: #6B46C1;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #6B46C1;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Checkbox */
    .stCheckbox > div {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Selectbox */
    .stSelectbox > div {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Text input */
    .stTextInput > div {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Number input */
    .stNumberInput > div {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to preprocess audio
def aggregate_2d(mfccs):
    return np.concatenate([
        np.mean(mfccs, axis=1),  
        np.std(mfccs, axis=1),   
        np.max(mfccs, axis=1),    
        np.min(mfccs, axis=1),    
    ])

def aggregate(feature):
    return np.array([
        np.mean(feature),
        np.std(feature),
        np.median(feature),
        np.max(feature),
        np.min(feature),
        np.percentile(feature, 25),  
        np.percentile(feature, 75)
    ])

def preprocess_wav(wav_file_path, sample_rate=16000):
    audio, sr = librosa.load(wav_file_path, sr=sample_rate)
    audio = librosa.effects.preemphasis(audio, coef=0.97)
    
    spectrogram = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))**2
    
    centroid = librosa.feature.spectral_centroid(S=spectrogram, sr=sr)
    centroid = aggregate(centroid)
   
    contrast = librosa.feature.spectral_contrast(S=spectrogram, sr=sr)
    contrast = aggregate_2d(contrast)
    
    flatness = librosa.feature.spectral_flatness(S=spectrogram)
    flatness = aggregate(flatness)
    
    rolloff = librosa.feature.spectral_rolloff(S=spectrogram, sr=sr)
    rolloff = aggregate(rolloff)

    mfccs = librosa.feature.mfcc(S=librosa.power_to_db(spectrogram), n_mfcc=13)
    mfccs = aggregate_2d(mfccs)
    
    features = np.concatenate([mfccs, centroid, contrast, flatness, rolloff],axis=0)
    return features

# Naive Bayes Classifier
def likelihood(x, mean, variance):
    return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(-0.5 * ((x - mean) ** 2) / variance)

class NaiveBayesClassifier():
    def __init__(self):
        self.variances= {0:{},1:{}}
        self.means= {0:{},1:{}}
        self.priors= {0:0,1:0}
    
    def fit(self,df):
        self.priors[0] = (df[df['label']==0].shape[0]) / df.shape[0]
        self.priors[1] = (df[df['label']==1].shape[0]) / df.shape[0]
        for i in range(len(df.columns)- 1):
            self.variances[0][i] = df[df['label']==0].iloc[:,i].var()
            self.variances[1][i] = df[df['label']==1].iloc[:,i].var()
            self.means[0][i] = df[df['label']==0].iloc[:,i].mean()
            self.means[1][i] = df[df['label']==1].iloc[:,i].mean()
    
    def predict_sample(self,x):
        likelihoods = {0:[],1:[]}
        for i in range(len(x)):
            likelihoods[0].append(likelihood(x[i],self.means[0][i],self.variances[0][i]))
            likelihoods[1].append(likelihood(x[i],self.means[1][i],self.variances[1][i]))
        posteriors = {0:0,1:0}
        for i in range(len(likelihoods[0])):
            posteriors[0] += np.log(likelihoods[0][i])
            posteriors[1] += np.log(likelihoods[1][i])
        posteriors[0] += np.log(self.priors[0])
        posteriors[1] += np.log(self.priors[1])
        if(posteriors[0] > posteriors[1]):
            return 0
        return 1

# Load the trained model
@st.cache_resource
def load_model():
    # Load the data
    df = pd.read_csv('data.csv')
    scaler = StandardScaler()
    X = scaler.fit_transform(df.drop(columns=['label']))
    y = df['label'].values
    df_scaled = pd.DataFrame(X, columns=df.columns[:-1])
    df_scaled['label'] = y
    
    # Train the model
    model = NaiveBayesClassifier()
    model.fit(df_scaled)
    return model, scaler

# Streamlit UI
st.markdown("<h1>🎤 Voice Gender Classifier</h1>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='color: #4A5568; font-size: 1.1rem;'>
            This app predicts gender (Male/Female) from voice recordings using a Naive Bayes classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Load model
model, scaler = load_model()

# Recording interface
st.markdown("<h2>🎙️ Record Your Voice</h2>", unsafe_allow_html=True)
st.markdown("""
    <div style='background-color: #F3F4F6; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;'>
        <p style='color: #4A5568; margin-bottom: 1rem;'>
            Click the button below to record your voice for 5 seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

if st.button("🎤 Start Recording"):
    fs = 44100
    duration = 5
    
    st.markdown("""
        <div style='background-color: #F0F9FF; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <p style='color: #6B46C1; text-align: center;'>Recording... Please speak now</p>
        </div>
        """, unsafe_allow_html=True)
    
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16',device=2)
    sd.wait()
    
    st.markdown("""
        <div style='background-color: #F0F9FF; padding: 1rem; border-radius: 8px; margin: 1rem 0;'>
            <p style='color: #6B46C1; text-align: center;'>Recording finished!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Save the recording
    write("temp_recording.wav", fs, audio)
    
    # Playback the recording
    st.markdown("<h3 style='color: #6B46C1;'>🎧 Playback Your Recording</h3>", unsafe_allow_html=True)
    st.audio("temp_recording.wav", format='audio/wav')
    
    # Process the recording
    features = preprocess_wav("temp_recording.wav")
    features = scaler.transform(features.reshape(1,-1))
    prediction = model.predict_sample(features.flatten())
    
    # Display result
    gender = "Male" if prediction == 0 else "Female"
    st.markdown(f"""
        <div class='success'>
            <h3 style='color: #6B46C1; margin: 0;'>Prediction Result</h3>
            <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Predicted Gender: <strong>{gender}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clean up
    os.remove("temp_recording.wav")

# File upload interface
st.markdown("<h2>📁 Upload a WAV File</h2>", unsafe_allow_html=True)
st.markdown("""
    <div style='background-color: #F3F4F6; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem;'>
        <p style='color: #4A5568; margin-bottom: 1rem;'>
            Or upload a WAV file for gender classification
        </p>
    </div>
    """, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a WAV file", type="wav")
if uploaded_file is not None:
    # Save the uploaded file
    with open("temp_upload.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Playback the uploaded file
    st.markdown("<h3 style='color: #6B46C1;'>🎧 Playback Uploaded Audio</h3>", unsafe_allow_html=True)
    st.audio("temp_upload.wav", format='audio/wav')
    
    # Process the file
    features = preprocess_wav("temp_upload.wav")
    features = scaler.transform(features.reshape(1,-1))
    prediction = model.predict_sample(features.flatten())
    
    # Display result
    gender = "Male" if prediction == 0 else "Female"
    st.markdown(f"""
        <div class='success'>
            <h3 style='color: #6B46C1; margin: 0;'>Prediction Result</h3>
            <p style='font-size: 1.2rem; margin: 0.5rem 0 0 0;'>Predicted Gender: <strong>{gender}</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Clean up
    os.remove("temp_upload.wav") 