import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import librosa
import sounddevice as sd
import pyaudio
import wave
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client
import os
import time
from datetime import datetime
import uuid
import csv
import geocoder
from dotenv import load_dotenv
import platform

# Load environment variables
load_dotenv()

# Configuration
SAMPLE_RATE = 16000  # YAMNet expects 16kHz
CHUNK = 1024  # Audio chunk size for streaming
DURATION = 10  # Record 10-second audio segments
DANGEROUS_CLASSES = [
    'Gunshot, gunfire', 'Scream', 'Glass break', 'Fire crackling', 'Explosion',
    'Shatter', 'Yell', 'Shout', 'Screaming', 'Fire', 'Glass', 'Breaking', 'Siren'
]
EMAIL_SENDER = os.getenv('EMAIL_SENDER', 'media270311@gmail.com')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD', 'sovy mmpn blxa jzpq')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT', 'modalasravanthi9390@gmail.com')
TWILIO_SID = os.getenv('TWILIO_SID', 'ACf2a9d28bbb58e7bba109504205f14319')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'e93dfd9717e39bd7671fd3e6f5713480')
TWILIO_FROM = os.getenv('TWILIO_FROM', '+13184889222')
TWILIO_TO = os.getenv('TWILIO_TO', '+919390717042')
CLASS_MAP_PATH = os.path.join(os.path.dirname(__file__), 'yamnet_class_map.csv')

# Check if running on Streamlit Cloud
IS_STREAMLIT_CLOUD = os.getenv('STREAMLIT_CLOUD') == 'true' or platform.system() == 'Linux'

# Load YAMNet model and class names
@st.cache_resource
def load_yamnet():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    if not os.path.exists(CLASS_MAP_PATH):
        st.error(f"Class map file not found at {CLASS_MAP_PATH}. Please include 'yamnet_class_map.csv' in the project root.")
        return None, None
    try:
        class_names = []
        with open(CLASS_MAP_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return model, class_names
    except Exception as e:
        st.error(f"Failed to parse class map file: {e}")
        return None, None

# Preprocess audio
def preprocess_audio(audio, sr=SAMPLE_RATE):
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) != 0 else audio
    target_length = SAMPLE_RATE * DURATION
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    return audio

# Save audio to WAV file
def save_audio(audio, filename, sr=SAMPLE_RATE):
    wavfile = wave.open(filename, 'wb')
    wavfile.setnchannels(1)
    wavfile.setsampwidth(2)  # 16-bit
    wavfile.setframerate(sr)
    wavfile.writeframes((audio * 32767).astype(np.int16).tobytes())
    wavfile.close()

# Send email alert
def send_email_alert(sound_class, audio_file, location):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECIPIENT
    msg['Subject'] = f'Dangerous Sound Detected: {sound_class}'
    body = f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} at {location}"
    msg.attach(MIMEText(body, 'plain'))
    with open(audio_file, 'rb') as f:
        part = MIMEText(f.read(), 'audio/wav', 'utf-8')
        part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(audio_file)}')
        msg.attach(part)
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Email sending failed: {e}")
        return False

# Send SMS alert
def send_sms_alert(sound_class, location):
    try:
        client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
        message = client.messages.create(
            body=f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} at {location}",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        return True
    except Exception as e:
        st.error(f"SMS sending failed: {e}")
        return False

# Record live audio with improved feedback
def record_audio():
    if IS_STREAMLIT_CLOUD:
        raise OSError("Live audio recording is not supported on Streamlit Cloud. Please upload a WAV file instead.")
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        st.write("Recording... (10 seconds)")
        progress_bar = st.progress(0)
        start_time = time.time()
        while time.time() - start_time < DURATION:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
            progress_bar.progress(min((time.time() - start_time) / DURATION, 1.0))
        stream.stop_stream()
        stream.close()
        p.terminate()
        audio = np.concatenate(frames)
        st.success("Recording completed!")
        return audio
    except OSError as e:
        p.terminate()
        st.error(f"Recording failed: {e}. Ensure a microphone is connected and permissions are granted.")
        return None

# Process audio and detect top four events
def process_audio(audio, audio_file=None):
    if audio is None:
        return []
    audio = preprocess_audio(audio)
    if audio_file:
        save_audio(audio, audio_file)
    scores, embeddings, spectrogram = model(audio)
    scores = scores.numpy()
    top_indices = np.argsort(scores[0])[::-1][:4]
    events = [(class_names[i], scores[0, i]) for i in top_indices]
    return events

# Streamlit app
def main():
    st.markdown("""
        <style>
        .stApp {
            background-color: #f0f0f5;
            color: #333333;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🔊 Real-Time Dangerous Sound Detection")
    st.write("Record or Upload Audio")

    global model, class_names
    with st.spinner("Loading model..."):
        model, class_names = load_yamnet()
    if model is None or class_names is None:
        return

    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = 0

    col1, col2 = st.columns([1, 2])
    with col1:
        if IS_STREAMLIT_CLOUD:
            st.info("Live audio recording is not supported on Streamlit Cloud. Please upload a WAV file instead.")
        else:
            if st.button("Record Live Audio (10 sec)"):
                with st.spinner("Initializing recording..."):
                    audio = record_audio()
                    if audio is not None:
                        audio_file = f"temp_audio_{uuid.uuid4()}.wav"
                        save_audio(audio, audio_file)
                        with open(audio_file, "rb") as f:
                            st.audio(f, format="audio/wav")
                        events = process_audio(audio, audio_file)
                        st.session_state.events = events
                        if os.path.exists(audio_file):
                            os.remove(audio_file)

    with col2:
        st.write("Or Upload a .wav Audio File")
        uploaded_file = st.file_uploader("Choose a WAV file", type="wav", accept_multiple_files=False, key="wav_uploader",
                                        help="Drag and drop file here. Limit 200MB per file - WAV")
        if uploaded_file:
            audio, sr = librosa.load(uploaded_file, sr=SAMPLE_RATE)
            audio_file = f"temp_{uuid.uuid4()}.wav"
            save_audio(audio, audio_file)
            st.audio(uploaded_file, format="audio/wav")
            events = process_audio(audio, audio_file)
            st.session_state.events = events
            if os.path.exists(audio_file):
                os.remove(audio_file)

    if 'events' in st.session_state:
        st.write("### Detected Events (with Confidence Scores)")
        for sound_class, confidence in st.session_state.events:
            is_dangerous = sound_class in DANGEROUS_CLASSES
            color = "#ff4d4d" if is_dangerous else "#4caf50"
            display_text = f"{sound_class} - Confidence: {confidence:.2%}"
            if is_dangerous:
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                            f"🔴 Danger Detected: {display_text}</div>", unsafe_allow_html=True)
                if time.time() - st.session_state.last_alert_time > 60:
                    st.session_state.last_alert_time = time.time()
                    audio_file = f"temp_alert_{uuid.uuid4()}.wav"
                    save_audio(audio, audio_file)
                    g = geocoder.ip('me')
                    location = g.city if g.ok else "Location not available"
                    email_sent = send_email_alert(sound_class, audio_file, location)
                    sms_sent = send_sms_alert(sound_class, location)
                    if email_sent and sms_sent:
                        st.markdown("<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px; color: #000000;'>"
                                    f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on "
                                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} at {location}</div>",
                                    unsafe_allow_html=True)
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
            else:
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                            f"✅ Safe sound: {display_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
