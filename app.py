import startup   # must be first
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

#embedder = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str

lemmatizer = WordNetLemmatizer()

# ---------------- Preprocess ----------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(lemmas)

# ---------------- Severity ----------------
def compute_severity(text, base):
    score = base
    boosters = [
        r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnothing\b",
        r"will fail", r"will never", r"by nature",
        r"\bdefinitely\b", r"\bmust\b", r"cant quit"
    ]
    for b in boosters:
        if re.search(b, text):
            score += 1
    return min(score, 5)

# ---------------- Rule Detectors ----------------
def detect_overgeneralization(text):
    return any(re.search(p, text) for p in [
        r"(fail|rude|bad|stupid|lazy).*(again|always|never)",
        r"(once|one time).*so.*(always|never)",
        r"(everyone|everybody).*so.*(always|never)"
    ])

def detect_bandwagon(text):
    return bool(re.search(r"(everyone|everybody|most people|all).*do", text))

def detect_confirmation_bias(text):
    return bool(re.search(r"(only|just).*trust|ignore.*other", text))

def detect_sunk_cost(text):
    return bool(re.search(r"(already|spend|put).*(time|money|effort)|cant quit|back out", text))

def detect_fundamental_attribution(text):
    return bool(re.search(r"(rude|lazy|stupid).*by nature|because.*is", text))

def detect_overconfidence(text):
    return bool(re.search(r"i am sure|definitely|never make mistake", text))

def detect_hindsight(text):
    return bool(re.search(r"i knew it|it was obvious", text))

# ---------------- Semantic Engine ----------------
semantic_templates = {
    "Sunk Cost Fallacy": [
        "I cannot quit because I have already invested too much",
        "I must continue because of past effort"
    ],
    "Overgeneralization Bias": [
        "One failure means I will always fail",
        "This happened once so it will happen every time"
    ],
    "Bandwagon Effect": [
        "Everyone is doing this so I should too"
    ],
    "Confirmation Bias": [
        "I only read information that supports my belief"
    ]
}

def semantic_detect(text):
    vec = embedder.encode([text])
    for bias, samples in semantic_templates.items():
        sample_vecs = embedder.encode(samples)
        scores = cosine_similarity(vec, sample_vecs)
        if scores.max() > 0.72:
            return bias
    return None

# ---------------- Analyze Route ----------------
@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    text = preprocess(data.text)

    if detect_sunk_cost(text):
        return {"bias":"Sunk Cost Fallacy","severity":compute_severity(text,3),
                "explanation":"You continue because of past effort.",
                "correction":"Base decisions on future benefit."}

    if detect_bandwagon(text):
        return {"bias":"Bandwagon Effect","severity":compute_severity(text,3),
                "explanation":"You follow others without independent judgement.",
                "correction":"Decide based on your own reasoning."}

    if detect_overgeneralization(text):
        return {"bias":"Overgeneralization Bias","severity":compute_severity(text,3),
                "explanation":"You generalize future from past failure.",
                "correction":"Each attempt is independent."}

    if detect_confirmation_bias(text):
        return {"bias":"Confirmation Bias","severity":compute_severity(text,3),
                "explanation":"You accept only belief-supporting info.",
                "correction":"Seek contradictory evidence."}

    if detect_fundamental_attribution(text):
        return {"bias":"Fundamental Attribution Error","severity":compute_severity(text,3),
                "explanation":"You blame personality not situation.",
                "correction":"Consider situational factors."}

    if detect_overconfidence(text):
        return {"bias":"Overconfidence Bias","severity":compute_severity(text,3),
                "explanation":"You overestimate your accuracy.",
                "correction":"Double-check assumptions."}

    if detect_hindsight(text):
        return {"bias":"Hindsight Bias","severity":compute_severity(text,2),
                "explanation":"Outcome seems obvious after it happens.",
                "correction":"Acknowledge uncertainty before events."}

    # 🔥 SEMANTIC FALLBACK LAYER (THIS WAS MISSING)
    semantic_bias = semantic_detect(data.text)
    if semantic_bias:
        return {
            "bias": semantic_bias,
            "severity": 4,
            "explanation": "Bias detected using semantic reasoning.",
            "correction": "Review the assumption objectively."
        }

    return {"bias":"No Bias Detected","severity":0,
            "explanation":"No cognitive bias pattern found.",
            "correction":"Your reasoning appears neutral."}

@app.get("/")
def home():
    return {"status":"Bias AI backend running successfully"}
