from collections import Counter

def compute_bias_trends(history):
    biases = [item["bias"] for item in history if item["bias"] != "No Bias Detected"]
    return dict(Counter(biases))
