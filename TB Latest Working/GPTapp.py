from PIL import Image
import os
import fitz  # PyMuPDF
from gtts import gTTS
#from googletrans import Translator
from deep_translator import GoogleTranslator
import google.generativeai as genai
import mimetypes

# --- API Key Configuration ---
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)
#translator = Translator()

# Language mapping for UI + gTTS
LANGUAGE_MAP = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa"
}

# --- Extract Text from Image ---
def extract_text_from_image(image_path):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()

        response = model.generate_content(
            contents=[
                {"mime_type": "image/jpeg", "data": img_data},
                "Extract all text from this image exactly as it appears."
            ]
        )
        return response.text.strip() if response.text else "No text extracted from image."
    except Exception as e:
        return f"Image OCR failed: {e}"

# --- Extract Text from PDF ---
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        return text.strip()
    except Exception as e:
        return f"PDF extraction failed: {e}"

# --- Generate Explanation ---
def explain_prescription(text):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Explain this prescription to a layman in simple terms.\n"
            "Do not change the names of the medicines.\n"
            "Clearly identify and label the DIAGNOSIS at the top.\n"
            "Then explain each medicine and give useful tips.\n"
            "NO bold or latex formatting. Plain text only."
        )
        response = model.generate_content([text, prompt])
        return response.text.strip() if response.text else "No explanation generated."
    except Exception as e:
        return f"Explanation failed: {e}"

# --- Translate Explanation ---
def translate_text(text, target_lang_code):
    if target_lang_code == "en":
        return text
    try:
        translated = GoogleTranslator(source="auto", target=target_lang_code).translate(text)
        return translated
    except Exception as e:
        return f"Translation error: {e}"


# --- Text-to-Speech ---
def text_to_speech(text, lang_code):
    try:
        audio_path = "output.mp3"
        tts = gTTS(text=text, lang=lang_code)
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        return f"TTS generation error: {e}"

# --- Core Function for Gradio Integration ---
def analyze_prescription_file(filepath, lang_code="en"):
    try:
        if filepath.endswith(".pdf") or mimetypes.guess_type(filepath)[0] == "application/pdf":
            extracted_text = extract_text_from_pdf(filepath)
        else:
            extracted_text = extract_text_from_image(filepath)

        explanation_en = explain_prescription(extracted_text)
        translated_explanation = translate_text(explanation_en, lang_code)
        audio_path = text_to_speech(translated_explanation, lang_code)

        # Extract diagnosis line
        diagnosis_line = next((line.strip() for line in explanation_en.splitlines() if "diagnosis" in line.lower()), "")
        translated_diagnosis = translate_text(diagnosis_line, lang_code) if diagnosis_line else "Diagnosis not found."

        return diagnosis_line, translated_explanation, audio_path
    except Exception as e:
        return "Error", f"Processing error: {e}", None
