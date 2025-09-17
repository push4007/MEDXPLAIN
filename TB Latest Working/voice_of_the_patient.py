# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

# Step1: Setup Audio recorder (ffmpeg & portaudio)
# ffmpeg, portaudio, pyaudio
import logging
import speech_recognition as sr
from pydub import AudioSegment
from io import BytesIO
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def record_audio(file_path, timeout=20, phrase_time_limit=None):
    """
    Simplified function to record audio from the microphone and save it as an MP3 file.

    Args:
    file_path (str): Path to save the recorded audio file.
    timeout (int): Maximum time to wait for a phrase to start (in seconds).
    phrase_time_limit (int): Maximum time for the phrase to be recorded (in seconds).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            logging.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logging.info("Start speaking now...")
            
            # Record the audio
            audio_data = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            logging.info("Recording complete.")
            
            # Convert the recorded audio to an MP3 file
            wav_data = audio_data.get_wav_data()
            audio_segment = AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3")
            
            logging.info(f"Audio saved to {file_path}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

audio_filepath = "patient_voice_test_for_patient.mp3"
# record_audio(file_path=audio_filepath)

# Step2: Setup Speech to text–STT–model for transcription
from groq import Groq

GROQ_API_KEY = "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"
#GROQ_API_KEY = "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"
stt_model = "whisper-large-v3"
#SARVAM_API_KEY = "cf8a2ecf-da18-4aa5-b990-bf884bf2cc0a"
SARVAM_API_KEY = "sk_jsjvvdcs_J3TX7fVA8FTHtojPeAsgT2q6"

def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    client = Groq(api_key=GROQ_API_KEY)
    
    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )
    return transcription.text

def transcribe_with_sarvam(audio_filepath):
    with open(audio_filepath, "rb") as f:
        files = {'file': (audio_filepath, f)}
        data = {
            'prompt': "convert text to english",
            'model': "saaras:v2",
        }
        headers = {
            "api-subscription-key": SARVAM_API_KEY
        }
        response = requests.post(
            "https://api.sarvam.ai/speech-to-text",
            headers=headers,
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        transcript = result.get("transcript")
        return transcript
    else:
        print(f"Error in Sarvam STT API call: {response.status_code} - {response.text}")
        return None
