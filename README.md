# Voice Gender Classification App

A Streamlit web application that predicts gender (Male/Female) from voice recordings using a Naive Bayes classifier.

## Features

- Record voice directly through the browser
- Upload WAV files for gender classification
- Audio playback functionality
- Modern and stylish UI with white and purple theme
- Real-time gender prediction

## Technical Details

The application uses:

- Librosa for audio processing
- Naive Bayes classifier for gender prediction
- Streamlit for the web interface
- Various audio features including MFCCs, spectral centroid, contrast, flatness, and rolloff

## Installation

1. Clone this repository:

```bash
git clone [your-repo-url]
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Usage

1. Record your voice:

   - Click the "Start Recording" button
   - Speak for 5 seconds
   - Listen to the playback
   - Get the gender prediction

2. Upload a WAV file:
   - Click the file uploader
   - Select a WAV file
   - Listen to the playback
   - Get the gender prediction

## Deployment

This app can be deployed on Streamlit Cloud for public access.

## License

MIT License
