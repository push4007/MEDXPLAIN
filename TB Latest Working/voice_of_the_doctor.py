# # if you don’t use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()
import requests
import os
import subprocess
import platform
import base64
from gtts import gTTS
from pydub import AudioSegment
# import elevenlabs
from elevenlabs.client import ElevenLabs
from sarvamai import SarvamAI

# ---------------------------
# Load API Keys from .env
# ---------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
# ---------------------------
# Utility: Play audio (cross-platform)
# ---------------------------
def play_audio(filepath):
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', filepath])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c',
                            f'(New-Object Media.SoundPlayer "{filepath}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', filepath])  # or mpg123/ffplay
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"Error playing audio: {e}")


# ---------------------------
# gTTS (Google Text-to-Speech)
# ---------------------------
def text_to_speech_with_gtts(input_text, output_filepath="gtts_output.mp3"):
    tts = gTTS(text=input_text, lang="en", slow=False)
    tts.save(output_filepath)
    play_audio(output_filepath)
    return output_filepath


# ---------------------------
# ElevenLabs TTS
# ---------------------------
def convert_mp3_to_wav(mp3_filepath, wav_filepath):
    audio = AudioSegment.from_mp3(mp3_filepath)
    audio.export(wav_filepath, format="wav")

def text_to_speech_with_elevenlabs(input_text, output_filepath_mp3="eleven_output.mp3"):
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.generate(
        text=input_text,
        voice="Aria",
        output_format="mp3_22050_32",
        model="eleven_turbo_v2"
    )
    elevenlabs.save(audio, output_filepath_mp3)

    # Convert to wav for playback
    output_filepath_wav = output_filepath_mp3.replace(".mp3", ".wav")
    convert_mp3_to_wav(output_filepath_mp3, output_filepath_wav)
    play_audio(output_filepath_wav)
    return output_filepath_wav


# ---------------------------
# SarvamAI TTS + Translate
# ---------------------------
sarvam_client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

def translate_text_with_sarvam(
    text: str,
    source_language_code: str,
    target_language_code: str
) -> str:
    """Call Sarvam’s Translate API and return the translated_text."""
    r = requests.post(
        "https://api.sarvam.ai/translate",
        headers={
            "api-subscription-key": "cf8a2ecf-da18-4aa5-b990-bf884bf2cc0a",
        },
        json={
            "input": text,
            "source_language_code": source_language_code,
            "target_language_code": target_language_code
        },
        timeout=10
    )
    r.raise_for_status()
    return r.json()["translated_text"]

def text_to_speech_with_sarvam(input_text, output_filepath_wav, language_code):

    if language_code != "en-IN":
        input_text = translate_text_with_sarvam(
            text=input_text,
            source_language_code="auto",
            target_language_code=language_code
        )
    response = requests.post(
    "https://api.sarvam.ai/text-to-speech", 
    headers={
        "api-subscription-key": "1246763d-8202-498e-b444-23810352380b"
    },
    json={
        "target_language_code": language_code,
        "text":input_text,
        "model": "bulbul:v2",
        "speaker": "anushka"
    },
)

# Save audio to file
    data = response.json()
    print("Full API response:", data)
    if not data.get("audios"):
        print("No audio returned.")
        return None

    # 2) Decode the first base64‑encoded WAV
    b64_audio = data["audios"][0]
    wav_bytes = base64.b64decode(b64_audio)

    with open(output_filepath_wav, "wb") as f:
        f.write(wav_bytes)
    print(f" Saved WAV to {output_filepath_wav}")
 
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath_wav])
        elif os_name == "Windows":  # Windows
            subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{output_filepath_wav}").PlaySync();'])
        elif os_name == "Linux":  # Linux
            subprocess.run(['aplay', output_filepath_wav])  # Alternative: use 'mpg123' or 'ffplay'
        else:
            raise OSError("Unsupported operating system")
    except Exception as e:
        print(f"An error occurred while trying to play the audio")
    
    return output_filepath_wav


# ---------------------------
