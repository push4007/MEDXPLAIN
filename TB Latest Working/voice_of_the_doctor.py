# if you don’t use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

import os
import subprocess
import platform
import base64
from gtts import gTTS
from pydub import AudioSegment
import elevenlabs
from elevenlabs.client import ElevenLabs
from sarvamai import SarvamAI

# ---------------------------
# Load API Keys from .env
# ---------------------------
ELEVENLABS_API_KEY = "sk_9045c45f8147630ce2d540ff73f6989378f5ec72b46e66a8" 
#SARVAM_API_KEY = "cf8a2ecf-da18-4aa5-b990-bf884bf2cc0a" 
SARVAM_API_KEY = "sk_jsjvvdcs_J3TX7fVA8FTHtojPeAsgT2q6"

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

def text_to_speech_with_sarvam(input_text, output_filepath_wav="sarvam_output.wav", language_code="en-IN"):
    # Call Sarvam TTS
    response = sarvam_client.text_to_speech.convert(
        text=input_text,
        target_language_code=language_code,
        model="bulbul:v2",
        speaker="anushka"
    )

    # Decode Base64 -> WAV
    if not response.audios:
        print("No audio returned from Sarvam.")
        return None

    b64_audio = response.audios[0]
    wav_bytes = base64.b64decode(b64_audio)

    with open(output_filepath_wav, "wb") as f:
        f.write(wav_bytes)
    print(f"Saved Sarvam WAV to {output_filepath_wav}")

    play_audio(output_filepath_wav)
    return output_filepath_wav


# ---------------------------
