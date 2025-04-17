###DangerousSoundDetector

DangerousSoundDetector is a Streamlit-based application for real-time detection of dangerous sounds (e.g., gunshots, screams, explosions) using the YAMNet model. It records or uploads audio, classifies sounds, and sends email/SMS alerts for dangerous events. The project also includes scripts to preprocess the UrbanSound8K dataset and train 1D-CNN, 2D-CNN, and LSTM models for audio classification.
Features

Real-Time Audio Detection: Record live audio or upload WAV files to detect dangerous sounds using YAMNet.
Alerts: Sends email and SMS alerts via Gmail and Twilio when dangerous sounds are detected.
Model Comparison: Displays accuracy metrics for YAMNet (90%), 1D-CNN (88%), 2D-CNN (92%), and LSTM (82%) based on ESC-50 dataset.
UrbanSound8K Support: Preprocess and train models on the UrbanSound8K dataset for 10-class audio classification.
Location Awareness: Includes IP-based location in alerts using geocoder.

###
Prerequisites

Python: Version 3.9 or higher.
UrbanSound8K Dataset: Download from UrbanSound8K and place at C:\Users\Sindhu\Desktop\IIT final\UrbanSound8K.
YAMNet Class Map: Ensure yamnet_class_map.csv is at C:\Users\Sindhu\Desktop\IIT final\yamnet_class_map.csv. Download from TensorFlow Hub if needed.
Accounts:
Gmail account for email alerts.
Twilio account for SMS alerts.



###
Account Creation
1. Gmail Account for Email Alerts

Create a Gmail Account (if you don’t have one):
Go to accounts.google.com.
Click "Create account" and follow the prompts to set up a new Gmail account (e.g., your_email@gmail.com).


###
Enable App Password (required for smtplib):
Enable 2-Step Verification:
Go to myaccount.google.com → Security → 2-Step Verification.
Follow the steps to enable it (e.g., via phone number).


###
Generate an App Password:
Go to Security → App passwords (search for "App passwords" if not visible).
Select "Mail" as the app and "Other" as the device, then name it (e.g., "DangerousSoundDetector").
Click "Generate" to get a 16-character password (e.g., xxxx xxxx xxxx xxxx).
Save this password securely; it will be used as EMAIL_PASSWORD.




Note:
EMAIL_SENDER: Your Gmail address (e.g., your_email@gmail.com).
EMAIL_RECIPIENT: The recipient’s email address (e.g., recipient_email@example.com).



###
2. Twilio Account for SMS Alerts

Create a Twilio Account:
Go to twilio.com.
Sign up with your email and create a password.
Verify your email and phone number as prompted.


Get Twilio Credentials:
Log in to console.twilio.com.
On the Dashboard, find:
Account SID: Your account identifier (e.g., ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx).
Auth Token: Your authentication token (e.g., xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx).
Save these as TWILIO_SID and TWILIO_AUTH_TOKEN.

Get a Twilio Phone Number:
Go to Phone Numbers → Buy a Number.
Purchase a phone number (e.g., +1XXXXXXXXXX) with SMS capabilities (free trial available).
Save this as TWILIO_FROM.

Set Recipient Number:
Choose a phone number to receive SMS alerts (e.g., +1YYYYYYYYYY).
Save this as TWILIO_TO.
Verify the recipient number in Twilio’s console if required (under Verified Caller IDs).

Note: Trial accounts require verified numbers for SMS. Upgrade to a paid account for unrestricted use.

###
Installation

Clone the Repository:
git clone https://github.com/yourusername/DangerousSoundDetector.git
cd DangerousSoundDetector

Replace yourusername with your GitHub username.

Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

###
Install Dependencies:
pip install -r requirements.txt
###
The requirements.txt includes:
streamlit==1.36.0
tensorflow==2.16.1
tensorflow-hub==0.16.1
numpy==1.26.4
librosa==0.10.2
sounddevice==0.4.7
pyaudio==0.2.14
twilio==9.2.3
geocoder==1.38.1

###
Verify Dataset and Class Map:

Ensure C:\Users\Sindhu\Desktop\IIT final\UrbanSound8K contains:
audio\fold1 to fold10 with WAV files.
metadata\UrbanSound8K.csv.


Ensure C:\Users\Sindhu\Desktop\IIT final\yamnet_class_map.csv exists.

###
Configuration
Edit app.py to set up credentials for email and SMS alerts:
EMAIL_SENDER = 'your_email@gmail.com'  # Your Gmail address
EMAIL_PASSWORD = 'your_app_password'   # Gmail App Password
EMAIL_RECIPIENT = 'recipient_email@example.com'  # Recipient email
TWILIO_SID = 'your_twilio_sid'        # Twilio Account SID
TWILIO_AUTH_TOKEN = 'your_twilio_auth_token'  # Twilio Auth Token
TWILIO_FROM = 'your_twilio_number'    # Twilio phone number (e.g., +1XXXXXXXXXX)
TWILIO_TO = 'recipient_phone_number'  # Recipient phone number (e.g., +1YYYYYYYYYY)

Usage
1. Run the Streamlit App
streamlit run app.py

###
Opens a web interface at http://localhost:8501.
Record Audio: Click "Record Live Audio (10 sec)" to capture 10 seconds of audio.
Upload Audio: Upload a WAV file to classify sounds.
Output: Displays detected events with YAMNet confidence scores. Dangerous sounds trigger email/SMS alerts with location.


###
2. Preprocess UrbanSound8K Data
python train_urbansound8k.py --output_dir "C:\Users\Sindhu\Desktop\IIT final\preprocessed_data"

###
Preprocesses audio files from C:\Users\Sindhu\Desktop\IIT final\UrbanSound8K\audio\fold1 to fold10.
Saves .npy files in C:\Users\Sindhu\Desktop\IIT final\preprocessed_data:
1D-CNN: Waveforms (train_0_audio.npy, shape: (16000, 1)).
2D-CNN: Spectrograms (train_0_spectrogram.npy, shape: (128, 100, 1)).
LSTM: MFCC sequences (train_0_mfcc.npy, shape: (4, 44, 13)).
###
3. Train Models
The same train_urbansound8k.py script trains 1D-CNN, 2D-CNN, and LSTM models:

Uses folds 1-8 for training, 9-10 for testing.
Saves trained models:
1dcnn_model.h5
2dcnn_model.h5
lstm_model.h5

###
Prints test accuracies (e.g., “1D-CNN Test Accuracy: 95.50%”).

Troubleshooting

Missing Files: If files like 99812-1-3-0.wav are missing, verify all WAV files are in UrbanSound8K\audio\fold*. The script skips missing files with error messages.
YAMNet Class Map: Ensure yamnet_class_map.csv is at the specified path.
Email Issues: Check Gmail App Password and 2-Step Verification settings.
SMS Issues: Verify Twilio credentials and ensure TWILIO_TO is a verified number in trial mode.
Dependencies: If installation fails, try pip install --upgrade pip or use a specific Python version (e.g., 3.9).

###
Project Structure
DangerousSoundDetector/
├── app.py                  # Streamlit app for real-time detection
├── train_urbansound8k.py   # Preprocess and train 1D-CNN, 2D-CNN, LSTM
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── preprocessed_data/      # Preprocessed UrbanSound8K data
│   ├── 1dcnn/
│   ├── 2dcnn/
│   ├── lstm/
├── 1dcnn_model.h5         # Trained 1D-CNN model
├── 2dcnn_model.h5         # Trained 2D-CNN model
├── lstm_model.h5          # Trained LSTM model

Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

