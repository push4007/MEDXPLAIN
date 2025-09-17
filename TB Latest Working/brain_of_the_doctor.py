# if you dont use pipenv uncomment the following:
# from dotenv import load_dotenv
# load_dotenv()

#Step1: Setup GROQ API key
import os

GROQ_API_KEY = "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"
#GROQ_API_KEY = "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"

#Step2: Convert image to required format
import base64

# image_path="acne.jpg"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

#Step3: Setup Multimodal LLM 
from groq import Groq

query = "Is there something wrong with my face?"
model = "meta-llama/llama-4-maverick-17b-128e-instruct"
# model="llama-3.2-90b-vision-preview" #Deprecated

def analyze_image_with_query(query: str, model: str, encoded_image: str = None):
    client = Groq(api_key=GROQ_API_KEY)
    if encoded_image:
        content = f"{query}\n\n[data:image/jpeg;base64,{encoded_image}]"
    else:
        content = query
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content
