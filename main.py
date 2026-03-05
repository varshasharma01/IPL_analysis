import base64
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()
client = Groq(api_key="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")

class ImageRequest(BaseModel):
    base64_image: str 


@app.post("/explain-chart")
async def explain_chart(request: ImageRequest):
    try:
        completion = client.chat.completions.create(
            
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Explain this IPL data chart. What are the key insights?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{request.base64_image}"}
                        }
                    ]
                }
            ]
        )
        return {"explanation": completion.choices[0].message.content}
    except Exception as e:
       
        print(f"Error detail: {e}") 
        raise HTTPException(status_code=500, detail=str(e))
    
# after clicking the button in the Streamlit app, it will send a POST request to this endpoint 
# with the base64 image, and the FastAPI server will process it and return the AI-generated explanation.
# this is happening in the background, so the user experience is seamless. 

