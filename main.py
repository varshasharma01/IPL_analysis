import os
import io
import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from ollama import Client  # Import the Ollama client

load_dotenv()
app = FastAPI()

# 1. Setup the Ollama Client with your specific IP and Port
OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL")
client = Client(
    host=OLLAMA_HOST,
    headers={"ngrok-skip-browser-warning": "true"}
)

class ImageRequest(BaseModel):
    base64_image: str

@app.post("/explain-chart")
async def explain_chart(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.base64_image)

        response = client.generate(
            model=OLLAMA_MODEL,
            prompt="You are an expert IPL data analyst. Explain this chart and provide the key insights in concise bullet points.",
            images=[image_bytes]
        )

        return {"explanation": response['response']}

    except Exception as e:
        print(f"Error detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))