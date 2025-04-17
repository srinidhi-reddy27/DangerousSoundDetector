# DangerousSoundDetector

**DangerousSoundDetector** is a Streamlit-based application for real-time detection of dangerous sounds (e.g., gunshots, screams, explosions) using the YAMNet model. It records or uploads audio, classifies sounds, and sends email/SMS alerts for dangerous events. The project also includes scripts to preprocess the UrbanSound8K dataset and train 1D-CNN, 2D-CNN, and LSTM models for audio classification.

## Features
- **Real-Time Audio Detection**: Record live audio or upload WAV files to detect dangerous sounds using YAMNet.
- **Alerts**: Sends email and SMS alerts via Gmail and Twilio when dangerous sounds are detected.
- **Model Comparison**: Displays accuracy metrics for YAMNet , 1D-CNN , 2D-CNN , and LSTM  based on UrbanSound8k dataset.
- **UrbanSound8K Support**: Preprocess and train models on the UrbanSound8K dataset for 10-class audio classification.
- **Location Awareness**: Includes IP-based location in alerts using `geocoder`.

## Prerequisites
- **Python**: Version 3.9 or higher.
- **UrbanSound8K Dataset**: Download from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) and place at `C:\Users\Sindhu\Desktop\IIT final\UrbanSound8K`.
- **YAMNet Class Map**: Ensure `yamnet_class_map.csv` is at `C:\Users\Sindhu\Desktop\IIT final\yamnet_class_map.csv`. Download from [TensorFlow Hub](https://tfhub.dev/google/yamnet/1) if needed.
- **Accounts**:
  - Gmail account for email alerts.
  - Twilio account for SMS alerts.

## Account Creation
### 1. Gmail Account for Email Alerts
1. **Create a Gmail Account** (if you don’t have one):
   - Go to [accounts.google.com](https://accounts.google.com).
   - Click "Create account" and follow the prompts to set up a new Gmail account (e.g., `your_email@gmail.com`).
2. **Enable App Password** (required for `smtplib`):
   - Enable 2-Step Verification:
     - Go to [myaccount.google.com](https://myaccount.google.com) → Security → 2-Step Verification.
     - Follow the steps to enable it (e.g., via phone number).
   - Generate an App Password:
     - Go to Security → App passwords (search for "App passwords" if not visible).
     - Select "Mail" as the app and "Other" as the device, then name it (e.g., "DangerousSoundDetector").
     - Click "Generate" to get a 16-character password (e.g., `xxxx xxxx xxxx xxxx`).
     - Save this password securely; it will be used as `EMAIL_PASSWORD`.
3. **Note**:
   - `EMAIL_SENDER`: Your Gmail address (e.g., `your_email@gmail.com`).
   - `EMAIL_RECIPIENT`: The recipient’s email address (e.g., `recipient_email@example.com`).

### 2. Twilio Account for SMS Alerts
1. **Create a Twilio Account**:
   - Go to [twilio.com](https://www.twilio.com/try-twilio).
   - Sign up with your email and create a password.
   - Verify your email and phone number as prompted.
2. **Get Twilio Credentials**:
   - Log in to [console.twilio.com](https://console.twilio.com).
   - On the Dashboard, find:
     - **Account SID**: Your account identifier (e.g., `ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
     - **Auth Token**: Your authentication token (e.g., `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`).
     - Save these as `TWILIO_SID` and `TWILIO_AUTH_TOKEN`.
3. **Get a Twilio Phone Number**:
   - Go to Phone Numbers → Buy a Number.
   - Purchase a phone number (e.g., `+1XXXXXXXXXX`) with SMS capabilities (free trial available).
   - Save this as `TWILIO_FROM`.
4. **Set Recipient Number**:
   - Choose a phone number to receive SMS alerts (e.g., `+1YYYYYYYYYY`).
   - Save this as `TWILIO_TO`.
   - Verify the recipient number in Twilio’s console if required (under Verified Caller IDs).
5. **Note**: Trial accounts require verified numbers for SMS. Upgrade to a paid account for unrestricted use.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/DangerousSoundDetector.git
   cd DangerousSoundDetector
