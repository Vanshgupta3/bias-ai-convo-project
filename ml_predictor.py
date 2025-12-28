# import torch
# from transformers import BertTokenizer, BertForSequenceClassification

# model = None
# tokenizer = None

# def get_model():
#     global model, tokenizer
#     if model is None:
#         print("Loading bias detection model...")
#         tokenizer = BertTokenizer.from_pretrained("bias_model")
#         model = BertForSequenceClassification.from_pretrained("bias_model")
#         model.eval()
#     return tokenizer, model

# def predict_bias(text):
#     tokenizer, model = get_model()
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     logits = outputs.logits
#     predicted = torch.argmax(logits, dim=1).item()
#     return predicted

import requests
import os

HF_API_URL = "https://api-inference.huggingface.co/models/YOUR_USERNAME/YOUR_MODEL_REPO"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def predict_bias(text):
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}"
    }
    payload = {"inputs": text}
    response = requests.post(HF_API_URL, headers=headers, json=payload)
    return response.json()
