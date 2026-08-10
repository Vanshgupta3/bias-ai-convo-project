import torch
from transformers import BertTokenizer, BertForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = BertTokenizer.from_pretrained("bias_model")
_model = BertForSequenceClassification.from_pretrained("bias_model")
_model.to(DEVICE)
_model.eval()


def predict_bias(text):
    """
    Returns (label: str, confidence: float 0-100) using the locally
    fine-tuned BERT model in bias_model/.
    """
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = int(torch.argmax(probs, dim=1).item())
    confidence = probs[0][pred_id].item() * 100  # scale to 0-100 to match bias_detector.py's threshold check

    label = _model.config.id2label[pred_id]
    return label, round(confidence, 2)