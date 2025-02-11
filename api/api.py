from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import joblib
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY")

REPO_ID = "harukidhruv/spam-detection-model"

try:
    model_path = hf_hub_download(repo_id=REPO_ID, filename="spam_detector_model.pkl")
    vectorizer_path = hf_hub_download(repo_id=REPO_ID, filename="tfidf_vectorizer.pkl")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    raise RuntimeError(f"Error loading model/vectorizer: {str(e)}")

class MessageRequest(BaseModel):
    message: str

class MessagesRequest(BaseModel):
    messages: List[str]

def validate_api_key(request: Request):
    api_key_in_request = request.headers.get("API-Key")
    if api_key_in_request != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid API key")

@app.get("/health")
def health_check(request: Request):
    validate_api_key(request)
    return {"status": "API is up and models are loaded"}

@app.post("/predict/")
def predict_spam(request: MessageRequest, request_info: Request):
    validate_api_key(request_info)
    try:
        vectorized_message = vectorizer.transform([request.message])
        prediction = model.predict(vectorized_message)
        label = "spam" if prediction[0] else "non-spam"
        return {"message": request.message, "label": label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch/")
def predict_batch_spam(request: MessagesRequest, request_info: Request):
    validate_api_key(request_info)
    try:
        vectorized_messages = vectorizer.transform(request.messages)
        predictions = model.predict(vectorized_messages)
        results = [{"message": msg, "label": "spam" if pred else "non-spam"} 
                   for msg, pred in zip(request.messages, predictions)]
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

