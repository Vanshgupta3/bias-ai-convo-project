import re
from ml_predictor import predict_bias

INTENSITY_WORDS = ["always", "never", "everyone", "nobody", "must", "completely"]
def compute_severity(text):
    score = 0
    words = text.lower().split()

    for w in words:
        if w in INTENSITY_WORDS:
            score += 15

    if len(text.split()) > 12:
        score += 10

    return min(100, max(30, score))

def detect_overgeneralization(text):
    patterns = [
        r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnobody\b",
        r"\bi fail(ed)?\b.*\ball\b", r"\bonce\b.*\balways\b"
    ]
    for p in patterns:
        if re.search(p, text.lower()):
            return True
    return False

def detect_sunk_cost(text):
    patterns = [
        r"\balready spent\b",
        r"\btoo much invested\b",
        r"\bcan't quit\b",
        r"\bcannot quit\b",
        r"\bso much time\b",
        r"\byears on this\b"
    ]
    for p in patterns:
        if re.search(p, text.lower()):
            return True
    return False


def analyze_bias(text):
    label, confidence = predict_bias(text)

    if confidence > 70 and label != "no_bias":
        return {
            "bias": label.replace("_", " ").title(),
            "severity": int(confidence),
            "explanation": f"Detected using ML model with {confidence}% confidence.",
            "correction": "Consider evaluating your reasoning from a neutral perspective."
        }

    if detect_sunk_cost(text):
        return {
            "bias": "Sunk Cost Fallacy",
            "severity": compute_severity(text),
            "explanation": "Past investment is forcing continuation even when it is irrational.",
            "correction": "Decide based on future benefit, not past effort."
        }

    if detect_overgeneralization(text):
        return {
            "bias": "Overgeneralization Bias",
            "severity": compute_severity(text),
            "explanation": "A single experience is being used to predict all future outcomes.",
            "correction": "Try evaluating your conclusion using multiple real examples instead of one event."
        }

    return {
        "bias": "No Bias Detected",
        "severity": 0,
        "explanation": "No clear thinking error found.",
        "correction": "Your reasoning appears neutral."
    }

