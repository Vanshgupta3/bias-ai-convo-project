import re
from ml_predictor import predict_bias

# =========================
# SEVERITY
# =========================
INTENSITY_WORDS = ["always", "never", "everyone", "nobody", "must", "completely"]


def compute_severity(text, base=30):
    score = base
    words = text.lower().split()

    for w in words:
        if w in INTENSITY_WORDS:
            score += 15

    if len(text.split()) > 12:
        score += 10

    return min(100, max(30, score))


# =========================
# RULE-BASED DETECTORS (fallback when BERT confidence is low)
# =========================
RULE_PATTERNS = {
    "Sunk Cost Fallacy": [
        r"\balready spent\b", r"\btoo much invested\b", r"\bcan't quit\b",
        r"\bcannot quit\b", r"\bso much time\b", r"\byears on this\b",
    ],
    "Overgeneralization Bias": [
        r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnobody\b",
        r"\bi fail(ed)?\b.*\ball\b", r"\bonce\b.*\balways\b",
    ],
    "Bandwagon Effect": [
        r"(everyone|most people|all).*do",
    ],
    "Confirmation Bias": [
        r"(only|just).*trust", r"ignore.*other",
    ],
    "Fundamental Attribution Error": [
        r"(lazy|stupid|rude).*by nature",
    ],
    "Overconfidence Bias": [
        r"i am sure", r"definitely", r"never wrong",
    ],
    "Hindsight Bias": [
        r"i knew it", r"it was obvious",
    ],
    "Availability Bias": [
        r"just saw", r"just heard", r"in the news",
    ],
}

EXPLANATIONS = {
    "Sunk Cost Fallacy": (
        "Past investment is forcing continuation even when it is irrational.",
        "Decide based on future benefit, not past effort.",
    ),
    "Overgeneralization Bias": (
        "A single experience is being used to predict all future outcomes.",
        "Try evaluating your conclusion using multiple real examples instead of one event.",
    ),
    "Bandwagon Effect": (
        "This judgment appears influenced by what most other people are doing.",
        "Decide based on your own independent reasoning.",
    ),
    "Confirmation Bias": (
        "Only information that supports an existing belief is being considered.",
        "Actively seek out evidence that could disprove your current belief.",
    ),
    "Fundamental Attribution Error": (
        "Someone's behavior is being attributed to their character rather than their situation.",
        "Consider situational factors that could explain the behavior.",
    ),
    "Overconfidence Bias": (
        "The certainty expressed here exceeds what the evidence supports.",
        "Re-examine the assumptions behind this certainty.",
    ),
    "Hindsight Bias": (
        "This outcome is being framed as though it was predictable, after the fact.",
        "Recall what was actually known and uncertain before the outcome happened.",
    ),
    "Availability Bias": (
        "Judgment here seems based on how easily examples come to mind, not actual likelihood.",
        "Check whether this is genuinely common or just memorable/recent.",
    ),
}


def rule_detect(text):
    """Return the first matching rule-based bias label, or None."""
    lowered = text.lower()
    for bias_name, patterns in RULE_PATTERNS.items():
        for p in patterns:
            if re.search(p, lowered):
                return bias_name
    return None


# =========================
# MAIN ANALYSIS FUNCTION
# =========================
def analyze_bias(text):
    label, confidence = predict_bias(text)

    if confidence > 70 and label != "no_bias":
        display_name = label.replace("_", " ").title()
        return {
            "bias": display_name,
            "severity": int(confidence),
            "confidence": confidence,
            "explanation": f"Detected using fine-tuned BERT model with {confidence}% confidence.",
            "correction": "Consider evaluating your reasoning from a neutral perspective.",
            "source": "ml_model",
        }

    # BERT wasn't confident enough — fall back to rule-based detection
    rule_bias = rule_detect(text)
    if rule_bias:
        explanation, correction = EXPLANATIONS[rule_bias]
        return {
            "bias": rule_bias,
            "severity": compute_severity(text),
            "confidence": confidence,
            "explanation": explanation,
            "correction": correction,
            "source": "rule_based",
        }

    return {
        "bias": "No Bias Detected",
        "severity": 0,
        "confidence": confidence,
        "explanation": "No clear thinking error found.",
        "correction": "Your reasoning appears neutral.",
        "source": "none",
    }