import gradio as gr
from PIL import Image
from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_patient import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts, text_to_speech_with_elevenlabs, text_to_speech_with_sarvam
from GPTapp import analyze_prescription_file, LANGUAGE_MAP
from xraymodel import load_onnx, predict_onnx, occlusion_heatmap, overlay_heatmap

system_prompt = """You have to act as a professional doctor, i know you are not but this is for learning purpose, talk like a doctor, do not talk like an ai agent"""

def process_inputs(audio_filepath, image_filepath, tts_engine, language_name):
    lang_code_tts = LANGUAGE_MAP.get(language_name, "en-IN")

    #lang_code_sarvam = language_name + "-IN" if "-IN" not in language_name else language_name

    SARVAM_LANGUAGE_MAP = {
    "English": "en-IN",
    "Hindi": "hi-IN",
    "Marathi": "mr-IN",
    "Bengali": "bn-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Gujrati": "gu-IN",
    "Odia": "or-IN",  # or "od" if that's what Sarvam expects
    "Punjabi": "pa-IN"
}

    lang_code_sarvam = SARVAM_LANGUAGE_MAP.get(language_name, "en")


    stt_output = transcribe_with_groq("whisper-large-v3" ,audio_filepath, "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")
    enc = encode_image(image_filepath) if image_filepath else None

    

    doctor_response = analyze_image_with_query(system_prompt + stt_output,model= "meta-llama/llama-4-maverick-17b-128e-instruct", encoded_image=enc)

    if tts_engine == "Sarvam AI":
        audio_path = text_to_speech_with_sarvam(doctor_response, "final.wav", language_code=lang_code_sarvam)
    elif tts_engine == "GTTS":
        audio_path = text_to_speech_with_gtts(doctor_response, "final.mp3", lang=lang_code_tts)
    else:
        audio_path = text_to_speech_with_elevenlabs(doctor_response, "final.mp3")

    return stt_output, doctor_response, audio_path

def prescription_analysis(file, language):
    lang_code = LANGUAGE_MAP.get(language, "en-IN")
    return analyze_prescription_file(file, lang_code)

def analyze_xray(image_path):
    image = Image.open(image_path).convert("RGB")
    sess = load_onnx("tuberModel.onnx")
    prob_tb, cls = predict_onnx(image, sess)
    label = "Tuberculosis" if cls == 1 else "Normal"
    heatmap = occlusion_heatmap(image, sess)
    vis_image = overlay_heatmap(image, heatmap)
    return f"{label} (TB probability = {prob_tb:.2f})", vis_image

with gr.Blocks() as demo:
    with gr.Tab("Voice Doctor"):
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Speak your symptoms")
        image_input = gr.Image(type="filepath", label="Upload Medical Image")
        tts_engine_dropdown = gr.Dropdown(["GTTS", "ElevenLabs", "Sarvam AI"], value="Sarvam AI", label="TTS Engine")
        language_dropdown = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="English", label="Output Language")
        stt_output = gr.Textbox(label="Speech to Text")
        diagnosis_output = gr.Textbox(label="Doctor's Response")
        audio_response = gr.Audio(label="Doctor's Voice", type="filepath")
        voice_btn = gr.Button("Submit")

        voice_btn.click(
            fn=process_inputs,
            inputs=[audio_input, image_input, tts_engine_dropdown, language_dropdown],
            outputs=[stt_output, diagnosis_output, audio_response]
        )

    with gr.Tab("X-ray TB Analyzer"):
        xray_input = gr.Image(type="filepath", label="Upload Chest X-ray")
        tb_label = gr.Textbox(label="Prediction")
        tb_heatmap = gr.Image(label="XAI Heatmap")
        xray_btn = gr.Button("Analyze X-ray")

        xray_btn.click(fn=analyze_xray, inputs=[xray_input], outputs=[tb_label, tb_heatmap])

    with gr.Tab("Prescription Analyzer"):
        file_input = gr.File(label="Upload Image or PDF")
        lang_select = gr.Dropdown(choices=list(LANGUAGE_MAP.keys()), value="English", label="Language")
        diagnosis_txt = gr.Textbox(label="Diagnosis")
        explanation_txt = gr.Textbox(label="Simplified Explanation")
        tts_output = gr.Audio(label="Audio Explanation", type="filepath")
        analyze_btn = gr.Button("Analyze")

        analyze_btn.click(
            fn=prescription_analysis,
            inputs=[file_input, lang_select],
            outputs=[diagnosis_txt, explanation_txt, tts_output]
        )

demo.launch(debug=True)
