# if you dont use pipenv uncomment the following:
from dotenv import load_dotenv
load_dotenv()

#VoiceBot UI with Gradio
import os
import gradio as gr
import requests
from io import BytesIO
from PIL import Image

from xraymodel import load_onnx, predict_onnx, occlusion_heatmap, overlay_heatmap
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import record_audio
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs
from GPTapp import analyze_prescription_file, LANGUAGE_MAP

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose. 
What's in this image?. Do you find anything wrong with it medically? 
If you make a differential, suggest some remedies for them. Donot add any numbers or special characters in 
your response. Your response should be in one long paragraph. Also always answer as if you are answering to a real person.
Donot say 'In the image I see' but say 'With what I see, I think you have ....'
Dont respond as an AI model in markdown, your answer should mimic that of an actual doctor not an AI bot, 
Keep your answer concise (max 2 sentences). No preamble, start your answer right away please"""


def transcribe_with_sarvam(audio_filepath, language_code="en"):
    """
    Transcribes audio using Sarvam API with language support.
    """
    url = "https://api.sarvam.ai/speech-to-text"

    headers = {
        "api-subscription-key": SARVAM_API_KEY
    }

    with open(audio_filepath, "rb") as f:
        files = {
            "file": (os.path.basename(audio_filepath), f, "audio/mpeg")
        }
        data = {
            "prompt": "convert text to english",  # You can customize prompt if needed
            "model": "saaras:v2",
            "language": language_code  # language code like 'en', 'hi', 'mr' etc.
        }

        response = requests.post(url, headers=headers, files=files, data=data)

    if response.status_code == 200:
        result = response.json()
        transcript = result.get("transcript", None)
        if transcript:
            return transcript
        else:
            raise Exception(f"Sarvam API did not return a transcript: {result}")
    else:
        raise Exception(f"Sarvam API error: {response.status_code} {response.text}")


def transcribe_with_groq(stt_model, audio_filepath, GROQ_API_KEY):
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    with open(audio_filepath, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=stt_model,
            file=audio_file,
            language="en"
        )

    return transcription.text


def process_inputs(audio_filepath, image_filepath, language):
    """
    Handles audio transcription (using Sarvam API), image analysis, and TTS.
    """
    lang_code = LANGUAGE_MAP.get(language, "en")

    # Use Sarvam for transcription with language support, fallback to Groq if needed
    try:
        speech_to_text_output = transcribe_with_sarvam(audio_filepath, language_code=lang_code)
    except Exception as e:
        print(f"Error during Sarvam transcription: {e}")
        # Fallback to Groq transcription
        speech_to_text_output = transcribe_with_groq(
            stt_model="whisper-large-v3",
            audio_filepath=audio_filepath,
            GROQ_API_KEY=GROQ_API_KEY
        )

    if image_filepath:
        doctor_response = analyze_image_with_query(
            query=system_prompt + speech_to_text_output,
            encoded_image=encode_image(image_filepath),
            model="llama-3.1-8b-instant"
        )
    else:
        doctor_response = "No image provided for me to analyze"

    wav_output_path = text_to_speech_with_elevenlabs(input_text=doctor_response, output_filepath_mp3="final.mp3")
    return speech_to_text_output, doctor_response, wav_output_path


def prescription_analysis(file, language):
    lang_code = LANGUAGE_MAP.get(language, "en")
    diagnosis, explanation, audio_path = analyze_prescription_file(file, lang_code)
    return diagnosis, explanation, audio_path


def analyze_xray(image_path):
    image = Image.open(image_path).convert("RGB")
    sess = load_onnx("tuberModel.onnx")
    prob_tb, cls = predict_onnx(image, sess)
    label = "Tuberculosis" if cls == 1 else "Normal"
    heatmap = occlusion_heatmap(image, sess)
    vis_image = overlay_heatmap(image, heatmap)
    return f"{label} (TB probability = {prob_tb:.2f})", vis_image


######################################
with gr.Blocks() as demo:
    with gr.Tab("Voice Doctor"):
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your symptoms")
        image_input = gr.Image(type="filepath", label="Upload Medical Image")
        language_dropdown = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="English", label="Language")
        stt_output = gr.Textbox(label="Speech to Text")
        diagnosis_output = gr.Textbox(label="Doctor's Response")
        audio_response = gr.Audio(label="Doctor's Voice", type="filepath")

        voice_btn = gr.Button("Submit")
        voice_btn.click(
            fn=process_inputs,
            inputs=[audio_input, image_input, language_dropdown],
            outputs=[stt_output, diagnosis_output, audio_response]
        )

    with gr.Tab("X-ray TB Analyzer"):
        xray_input = gr.Image(type="filepath", label="Upload Chest X-ray")
        tb_label = gr.Textbox(label="Prediction")
        tb_heatmap = gr.Image(label="XAI Heatmap")

        xray_btn = gr.Button("Analyze X-ray")
        xray_btn.click(
            fn=analyze_xray,
            inputs=[xray_input],
            outputs=[tb_label, tb_heatmap]
        )

    with gr.Tab("Prescription Analyzer"):
        file_input = gr.File(label="Upload Image or PDF")
        language_dropdown2 = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="English", label="Language")
        diagnosis_txt = gr.Textbox(label="Diagnosis")
        explanation_txt = gr.Textbox(label="Simplified Explanation")
        tts_output = gr.Audio(label="Audio Explanation", type="filepath")

        analyze_btn = gr.Button("Analyze")
        analyze_btn.click(
            fn=prescription_analysis,
            inputs=[file_input, language_dropdown2],
            outputs=[diagnosis_txt, explanation_txt, tts_output]
        )

demo.launch(debug=True)

# http://127.0.0.1:7860
