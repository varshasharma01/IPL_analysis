import os
import io
import base64
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from google import genai
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from google import genai
import requests

load_dotenv() #
app = FastAPI()

# Get the key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)

class ImageRequest(BaseModel):
    base64_image: str

@app.post("/explain-chart")
async def explain_chart(request: ImageRequest, authorization: str = Header(None)):
    # Security check (matching the key sent from Streamlit)
    if not authorization or authorization != f"Bearer {GEMINI_API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    try:
        # 1. Decode the base64 string back into an image
        image_data = base64.b64decode(request.base64_image)
        img = Image.open(io.BytesIO(image_data))

        # 2. Call Gemini 1.5 Flash (optimized for vision/charts)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[
                "You are an expert IPL data analyst. Explain this chart and provide the key insights in concise bullet points.",
                img
            ]
        )

        return {"explanation": response.text}

    except Exception as e:
        print(f"Error detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))