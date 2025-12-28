import torch
from transformers import BertTokenizer, BertForSequenceClassification

model = None
tokenizer = None

def get_model():
    global model, tokenizer
    if model is None:
        print("Loading bias detection model...")
        tokenizer = BertTokenizer.from_pretrained("bias_model")
        model = BertForSequenceClassification.from_pretrained("bias_model")
        model.eval()
    return tokenizer, model

def predict_bias(text):
    tokenizer, model = get_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted = torch.argmax(logits, dim=1).item()
    return predicted
