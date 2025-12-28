import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bias_model")
model = BertForSequenceClassification.from_pretrained("bias_model")
model.eval()

LABELS = ["availability", "no_bias", "overgeneralization", "sunk_cost"]

def predict_bias(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1)
    score, label_id = torch.max(probs, dim=1)
    return LABELS[label_id.item()], round(score.item() * 100, 2)
