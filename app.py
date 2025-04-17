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
import geocoder  # Added for location

# Configuration
SAMPLE_RATE = 16000  # YAMNet expects 16kHz
CHUNK = 1024  # Audio chunk size for streaming
DURATION = 10  # Record 10-second audio segments
DANGEROUS_CLASSES = [
    'Gunshot, gunfire', 'Scream', 'Glass break', 'Fire crackling', 'Explosion', 
    'Shatter', 'Yell', 'Shout', 'Screaming', 'Fire','Glass','Breaking','Siren'
]  # YAMNet classes considered dangerous
EMAIL_SENDER = 'media270311@gmail.com'  # Replace with your Gmail
EMAIL_PASSWORD = 'sovy mmpn blxa jzpq'  # Replace with Gmail app password
EMAIL_RECIPIENT = 'srinidhigouragari7@gmail.com'  # Replace with recipient email
TWILIO_SID = 'AC3773b3d45f9effb0c07938c8a3b45b47'  # Replace with Twilio SID
TWILIO_AUTH_TOKEN = 'c4d15eef9430b86e0a0407aac588dd61'  # Replace with Twilio Auth Token
TWILIO_FROM = '+19475006677'  # Replace with Twilio phone number
TWILIO_TO = '+917729936696'
 # Replace with recipient phone number
CLASS_MAP_PATH = r"yamnet_class_map.csv"  # Local path to class map

# Load YAMNet model and class names
@st.cache_resource
def load_yamnet():
    model = hub.load('https://tfhub.dev/google/yamnet/1')
    try:
        class_names = []
        with open(CLASS_MAP_PATH, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                class_names.append(row['display_name'])
        return model, class_names
    except FileNotFoundError:
        st.error(f"Class map file not found at {CLASS_MAP_PATH}. Please ensure the file exists.")
        return None, None
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
    body = f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
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
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    try:
        message = client.messages.create(
            body=f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} at {location}",
            from_=TWILIO_FROM,
            to=TWILIO_TO
        )
        return True
    except Exception as e:
        st.error(f"SMS sending failed: {e}")
        return False

# Record live audio
def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    st.write("Recording... (10 seconds)")
    start_time = time.time()
    while time.time() - start_time < DURATION:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0)
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio = np.concatenate(frames)
    return audio

# Process audio and detect top four events
def process_audio(audio, audio_file=None):
    audio = preprocess_audio(audio)
    if audio_file:
        save_audio(audio, audio_file)
    scores, embeddings, spectrogram = model(audio)
    scores = scores.numpy()
    # Get indices of top 4 scores
    top_indices = np.argsort(scores[0])[::-1][:4]
    events = [(class_names[i], scores[0, i]) for i in top_indices]
    return events

# Streamlit app
def main():
    # Set light background
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

    # Load model
    global model, class_names
    with st.spinner("Loading model..."):
        model, class_names = load_yamnet()
    if model is None or class_names is None:
        return
    # Session state
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = 0

    # Record or Upload options
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Record Live Audio (10 sec)"):
            with st.spinner("Recording..."):
                audio = record_audio()
                audio_file = f"temp_audio_{uuid.uuid4()}.wav"
                save_audio(audio, audio_file)
                # Display the recorded audio
                with open(audio_file, "rb") as f:
                    st.audio(f, format="audio/wav")
                # Process the audio for event detection
                events = process_audio(audio, audio_file)
                st.session_state.events = events
                # Clean up the temporary audio file
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

    # Display detected events inline with messages, emphasizing confidence
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
                    # Get location using geocoder (IP-based, approximate)
                    g = geocoder.ip('me')
                    location = g.city if g.ok else "Location not available"
                    email_sent = send_email_alert(sound_class, audio_file, location)
                    sms_sent = send_sms_alert(sound_class, location)
                    if email_sent and sms_sent:
                        st.markdown("<div style='padding: 10px; border: 1px solid #ccc; border-radius: 5px; color: #000000;'>"
                                    f"🌐 Alert Sent\nMessage: Danger Detected: {sound_class} on "
                                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", 
                                    unsafe_allow_html=True)
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
            else:
                st.markdown(f"<div style='background-color: {color}; padding: 10px; border-radius: 5px;'>"
                            f"✅ Safe sound: {display_text}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
