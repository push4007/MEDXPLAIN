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
client = SarvamAI(api_subscription_key=SARVAM_API_KEY)

def chunk_text(text, max_len=2000):
    """Split text into safe chunks for Sarvam."""
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def translate_text_with_sarvam(text, source_language_code="en-IN", target_language_code="hi-IN"):
    """Translate text safely with Sarvam API."""
    chunks = chunk_text(text)
    translated_chunks = []

    for chunk in chunks:
        response = client.text.translate(
            input=chunk,
            source_language_code=source_language_code,
            target_language_code=target_language_code,
            mode="formal",
            model="sarvam-translate:v1",
            numerals_format="native",
            speaker_gender="Male",
            enable_preprocessing=False
        )
        # ✅ Access attribute, not dict key
        translated_chunks.append(response.translated_text)

    return " ".join(translated_chunks)



def text_to_speech_with_sarvam(input_text: str, output_filepath_wav: str, language_code: str):
    """Convert text to speech using Sarvam TTS and save as .wav file."""
    # Translate if language is not English
    if language_code != "en-IN":
        input_text = translate_text_with_sarvam(
            text=input_text,
            source_language_code="en-IN",
            target_language_code=language_code
        )

    response = client.text_to_speech.convert(
        text=input_text,
        target_language_code=language_code,
        model="bulbul:v1",
        speaker="maya"
    )

    # Debug print
    print("Full API response:", response)

    audios = response.audios or []
    if not audios:
        print("No audio returned.")
        return None

    # Decode base64-encoded WAV
    wav_bytes = base64.b64decode(audios[0])
    with open(output_filepath_wav, "wb") as f:
        f.write(wav_bytes)

    print(f"✅ Saved WAV to {output_filepath_wav}")

    # Try playing audio cross-platform
    os_name = platform.system()
    try:
        if os_name == "Darwin":  # macOS
            subprocess.run(['afplay', output_filepath_wav])
        elif os_name == "Windows":
            subprocess.run(['powershell', '-c',
                            f'(New-Object Media.SoundPlayer "{output_filepath_wav}").PlaySync();'])
        elif os_name == "Linux":
            subprocess.run(['aplay', output_filepath_wav])
        else:
            raise OSError("Unsupported OS")
    except Exception:
        print("⚠️ Could not play audio automatically")

    return output_filepath_wav

# ---------------------------
