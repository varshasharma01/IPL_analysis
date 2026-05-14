import os
import base64
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

class ImageRequest(BaseModel):
    base64_image: str

@app.post("/explain-chart")
async def explain_chart(request: ImageRequest):
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{MODEL}",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                },
                json={
                    "inputs": {
                        "image": request.base64_image,
                        "text": "You are an expert IPL data analyst. Explain this chart and provide key insights in concise bullet points."
                    }
                }
            )
            result = response.json()
            return {"explanation": result[0].get("generated_text", str(result))}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))