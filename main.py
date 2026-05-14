import os
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from ollama import Client  # Import the Ollama client
import requests

load_dotenv()
app = FastAPI()
HF_TOKEN = os.environ.get("HF_TOKEN")  # from huggingface.co/settings/tokens
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

def explain_chart(image_bytes):
    img_b64 = base64.b64encode(image_bytes).decode()
    response = requests.post(
        f"https://api-inference.huggingface.co/models/{MODEL}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={
            "inputs": {
                "image": img_b64,
                "text": "You are an IPL data analyst. Explain this chart with key insights."
            }
        }
    )
    return response.json()