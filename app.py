# # import startup   # must be first
# # from fastapi import FastAPI
# # from pydantic import BaseModel
# # from fastapi.middleware.cors import CORSMiddleware
# # import re
# # from nltk.stem import WordNetLemmatizer
# # from nltk.tokenize import word_tokenize
# # #from sentence_transformers import SentenceTransformer
# # from sklearn.metrics.pairwise import cosine_similarity

# # #embedder = SentenceTransformer("all-MiniLM-L6-v2")

# # app = FastAPI()

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )

# # class AnalyzeRequest(BaseModel):
# #     text: str

# # lemmatizer = WordNetLemmatizer()

# # # ---------------- Preprocess ----------------
# # def preprocess(text):
# #     text = text.lower()
# #     text = re.sub(r"[^a-z\s]", "", text)
# #     tokens = word_tokenize(text)
# #     lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
# #     return " ".join(lemmas)

# # # ---------------- Severity ----------------
# # def compute_severity(text, base):
# #     score = base
# #     boosters = [
# #         r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnothing\b",
# #         r"will fail", r"will never", r"by nature",
# #         r"\bdefinitely\b", r"\bmust\b", r"cant quit"
# #     ]
# #     for b in boosters:
# #         if re.search(b, text):
# #             score += 1
# #     return min(score, 5)

# # # ---------------- Rule Detectors ----------------
# # def detect_overgeneralization(text):
# #     return any(re.search(p, text) for p in [
# #         r"(fail|rude|bad|stupid|lazy).*(again|always|never)",
# #         r"(once|one time).*so.*(always|never)",
# #         r"(everyone|everybody).*so.*(always|never)"
# #     ])

# # def detect_bandwagon(text):
# #     return bool(re.search(r"(everyone|everybody|most people|all).*do", text))

# # def detect_confirmation_bias(text):
# #     return bool(re.search(r"(only|just).*trust|ignore.*other", text))

# # def detect_sunk_cost(text):
# #     return bool(re.search(r"(already|spend|put).*(time|money|effort)|cant quit|back out", text))

# # def detect_fundamental_attribution(text):
# #     return bool(re.search(r"(rude|lazy|stupid).*by nature|because.*is", text))

# # def detect_overconfidence(text):
# #     return bool(re.search(r"i am sure|definitely|never make mistake", text))

# # def detect_hindsight(text):
# #     return bool(re.search(r"i knew it|it was obvious", text))

# # # ---------------- Semantic Engine ----------------
# # semantic_templates = {
# #     "Sunk Cost Fallacy": [
# #         "I cannot quit because I have already invested too much",
# #         "I must continue because of past effort"
# #     ],
# #     "Overgeneralization Bias": [
# #         "One failure means I will always fail",
# #         "This happened once so it will happen every time"
# #     ],
# #     "Bandwagon Effect": [
# #         "Everyone is doing this so I should too"
# #     ],
# #     "Confirmation Bias": [
# #         "I only read information that supports my belief"
# #     ]
# # }

# # def semantic_detect(text):
# #     vec = embedder.encode([text])
# #     for bias, samples in semantic_templates.items():
# #         sample_vecs = embedder.encode(samples)
# #         scores = cosine_similarity(vec, sample_vecs)
# #         if scores.max() > 0.72:
# #             return bias
# #     return None

# # # ---------------- Analyze Route ----------------
# # @app.post("/analyze")
# # def analyze(data: AnalyzeRequest):
# #     text = preprocess(data.text)

# #     if detect_sunk_cost(text):
# #         return {"bias":"Sunk Cost Fallacy","severity":compute_severity(text,3),
# #                 "explanation":"You continue because of past effort.",
# #                 "correction":"Base decisions on future benefit."}

# #     if detect_bandwagon(text):
# #         return {"bias":"Bandwagon Effect","severity":compute_severity(text,3),
# #                 "explanation":"You follow others without independent judgement.",
# #                 "correction":"Decide based on your own reasoning."}

# #     if detect_overgeneralization(text):
# #         return {"bias":"Overgeneralization Bias","severity":compute_severity(text,3),
# #                 "explanation":"You generalize future from past failure.",
# #                 "correction":"Each attempt is independent."}

# #     if detect_confirmation_bias(text):
# #         return {"bias":"Confirmation Bias","severity":compute_severity(text,3),
# #                 "explanation":"You accept only belief-supporting info.",
# #                 "correction":"Seek contradictory evidence."}

# #     if detect_fundamental_attribution(text):
# #         return {"bias":"Fundamental Attribution Error","severity":compute_severity(text,3),
# #                 "explanation":"You blame personality not situation.",
# #                 "correction":"Consider situational factors."}

# #     if detect_overconfidence(text):
# #         return {"bias":"Overconfidence Bias","severity":compute_severity(text,3),
# #                 "explanation":"You overestimate your accuracy.",
# #                 "correction":"Double-check assumptions."}

# #     if detect_hindsight(text):
# #         return {"bias":"Hindsight Bias","severity":compute_severity(text,2),
# #                 "explanation":"Outcome seems obvious after it happens.",
# #                 "correction":"Acknowledge uncertainty before events."}

# #     # 🔥 SEMANTIC FALLBACK LAYER (THIS WAS MISSING)
# #     semantic_bias = semantic_detect(data.text)
# #     if semantic_bias:
# #         return {
# #             "bias": semantic_bias,
# #             "severity": 4,
# #             "explanation": "Bias detected using semantic reasoning.",
# #             "correction": "Review the assumption objectively."
# #         }

# #     return {"bias":"No Bias Detected","severity":0,
# #             "explanation":"No cognitive bias pattern found.",
# #             "correction":"Your reasoning appears neutral."}

# # @app.get("/")
# # def home():
# #     return {"status":"Bias AI backend running successfully"}
# import startup   # must be first

# from fastapi import FastAPI
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware

# import re
# import torch
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize

# from transformers import BertTokenizer, BertForSequenceClassification
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# =========================
# FASTAPI SETUP
# =========================
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # =========================
# # REQUEST SCHEMA
# # =========================
# class AnalyzeRequest(BaseModel):
#     text: str

# # =========================
# # PREPROCESSING
# # =========================
# lemmatizer = WordNetLemmatizer()

# def preprocess(text):
#     text = text.lower()
#     text = re.sub(r"[^a-z\s]", "", text)
#     tokens = word_tokenize(text)
#     lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
#     return " ".join(lemmas)

# # =========================
# # SEVERITY
# # =========================
# def compute_severity(text, base):
#     score = base
#     boosters = [
#         r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnothing\b",
#         r"will fail", r"will never", r"by nature",
#         r"\bdefinitely\b", r"\bmust\b", r"cant quit"
#     ]
#     for b in boosters:
#         if re.search(b, text):
#             score += 1
#     return min(score, 5)

# # =========================
# # RULE-BASED DETECTORS
# # =========================
# def detect_overgeneralization(text):
#     return any(re.search(p, text) for p in [
#         r"(fail|bad|stupid|lazy).*(again|always|never)",
#         r"(once|one time).*so.*(always|never)"
#     ])
# # def detect_overgeneralization(text):
# #     patterns = [
# #         # existing patterns
# #         r"(fail|rude|bad|stupid|lazy).*(again|always|never)",
# #         r"(once|one time).*so.*(always|never)",
# #         r"(everyone|everybody).*so.*(always|never)",

# #         # 🔥 NEW REAL-WORLD PATTERNS (ADD THESE)
# #         r"(once|one experience).*?(so|therefore).*?(generation|everyone|all)",
# #         r"(one person|one event).*?means.*?(everyone|all|generation)",
# #         r"(cheated|betrayed).*?(so|therefore).*?(everyone|all|generation)",
# #         r"(one relationship|one incident).*?(means|proves).*?(love|loyalty).*(doesnt|does not|isnt|is not)"
# #     ]

# #     return any(re.search(p, text) for p in patterns)

# def detect_bandwagon(text):
#     return bool(re.search(r"(everyone|everybody|most people|all).*do", text))

# def detect_confirmation_bias(text):
#     return bool(re.search(r"(only|just).*trust|ignore.*other", text))

# def detect_sunk_cost(text):
#     return bool(re.search(r"(already|spent|put).*(time|money|effort)|cant quit|back out", text))

# def detect_fundamental_attribution(text):
#     return bool(re.search(r"(rude|lazy|stupid).*by nature", text))

# def detect_overconfidence(text):
#     return bool(re.search(r"i am sure|definitely|never make mistake", text))

# def detect_hindsight(text):
#     return bool(re.search(r"i knew it|it was obvious", text))

# # =========================
# # SEMANTIC ENGINE
# # =========================
# embedder = SentenceTransformer("all-MiniLM-L6-v2")

# semantic_templates = {
#     "Sunk Cost Fallacy": [
#         "I cannot quit because I have already invested too much",
#         "Stopping now would waste all my effort"
#     ],
#     "Overgeneralization Bias": [
#         "One failure means I will always fail",
#         "Past failure decides my future"
#     ],
#     "Bandwagon Effect": [
#         "Everyone is doing this so I should too"
#     ],
#     "Confirmation Bias": [
#         "I only read information that supports my belief"
#     ]
# }

# def semantic_detect(text):
#     vec = embedder.encode([text])
#     for bias, samples in semantic_templates.items():
#         sample_vecs = embedder.encode(samples)
#         scores = cosine_similarity(vec, sample_vecs)
#         if scores.max() > 0.70:
#             return bias
#     return None

# # =========================
# # LOAD TRAINED BERT
# # =========================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# bert_tokenizer = BertTokenizer.from_pretrained("bias_model")
# bert_model = BertForSequenceClassification.from_pretrained("bias_model")
# bert_model.to(DEVICE)
# bert_model.eval()

# def bert_detect(text):
#     inputs = bert_tokenizer(
#         text,
#         return_tensors="pt",
#         padding=True,
#         truncation=True,
#         max_length=128
#     )
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = bert_model(**inputs)

#     logits = outputs.logits
#     pred_id = torch.argmax(logits, dim=1).item()
#     confidence = torch.softmax(logits, dim=1)[0][pred_id].item()
#     label = bert_model.config.id2label[pred_id]

#     return label, confidence

# # =========================
# # ANALYZE ROUTE
# # =========================
# @app.post("/analyze")
# def analyze(data: AnalyzeRequest):
#     clean_text = preprocess(data.text)

#     if detect_sunk_cost(clean_text):
#         return {"bias":"Sunk Cost Fallacy","severity":compute_severity(clean_text,3),
#                 "explanation":"You continue because of past investment.",
#                 "correction":"Base decisions on future benefit."}

#     if detect_bandwagon(clean_text):
#         return {"bias":"Bandwagon Effect","severity":compute_severity(clean_text,3),
#                 "explanation":"You follow others instead of reasoning independently.",
#                 "correction":"Decide based on your own judgement."}

#     if detect_overgeneralization(clean_text):
#         return {"bias":"Overgeneralization Bias","severity":compute_severity(clean_text,3),
#                 "explanation":"You generalize future outcomes from limited events.",
#                 "correction":"One event does not define all outcomes."}

#     if detect_confirmation_bias(clean_text):
#         return {"bias":"Confirmation Bias","severity":compute_severity(clean_text,3),
#                 "explanation":"You accept only information that supports your belief.",
#                 "correction":"Seek opposing evidence."}

#     if detect_fundamental_attribution(clean_text):
#         return {"bias":"Fundamental Attribution Error","severity":compute_severity(clean_text,3),
#                 "explanation":"You blame personality instead of situation.",
#                 "correction":"Consider context."}

#     if detect_overconfidence(clean_text):
#         return {"bias":"Overconfidence Bias","severity":compute_severity(clean_text,3),
#                 "explanation":"You overestimate your accuracy.",
#                 "correction":"Double-check assumptions."}

#     if detect_hindsight(clean_text):
#         return {"bias":"Hindsight Bias","severity":compute_severity(clean_text,2),
#                 "explanation":"Outcome seems obvious after it happens.",
#                 "correction":"Recognize uncertainty before events."}

#     # ---------- SEMANTIC FALLBACK ----------
#     semantic_bias = semantic_detect(data.text)
#     if semantic_bias:
#         return {
#             "bias": semantic_bias,
#             "severity": 4,
#             "explanation": "Bias detected using semantic reasoning.",
#             "correction": "Review the assumption objectively."
#         }

#     # ---------- FINAL BERT FALLBACK ----------
#     bert_bias, bert_conf = bert_detect(data.text)
#     if bert_conf > 0.50:
#         return {
#             "bias": bert_bias,
#             "severity": 4,
#             "confidence": round(bert_conf, 2),
#             "explanation": "Bias detected using transformer-based BERT model.",
#             "correction": "Re-evaluate this assumption objectively."
#         }

#     return {
#         "bias":"No Bias Detected",
#         "severity":0,
#         "explanation":"No cognitive bias pattern found.",
#         "correction":"Your reasoning appears neutral."
#     }

# # =========================
# # HEALTH CHECK
# # =========================
# @app.get("/")
# def home():
#     return {"status":"Bias AI backend running successfully"}

import startup  # must be first

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import re
import torch
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification

# =========================
# FASTAPI SETUP
# =========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# REQUEST SCHEMA
# =========================
class AnalyzeRequest(BaseModel):
    text: str

# =========================
# PREPROCESSING
# =========================
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(tok) for tok in tokens]
    return " ".join(lemmas)

# =========================
# SEVERITY
# =========================
def compute_severity(text, base):
    boosters = [r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bnothing\b",
                r"definitely", r"\bmust\b"]
    for b in boosters:
        if re.search(b, text):
            base += 1
    return min(base, 5)

# =========================
# RULE DETECTORS
# =========================
def detect_overgeneralization(text):
    patterns = [
        r"once.*(so|therefore|hence).*generation",
        r"loyalty is extinct",
        r"people these days",
        r"everyone is like this",
        r"nobody.*anymore"
    ]
    return any(re.search(p, text) for p in patterns)

def detect_sunk_cost(text):
    patterns = [
        r"worked hard",
        r"cant say no",
        r"after all i did",
        r"waste if i stop",
        r"too much invested",
        r"so much effort",
        r"past effort"
    ]
    return any(re.search(p, text) for p in patterns)

def detect_bandwagon(text):
    return bool(re.search(r"(everyone|most people|all).*do", text))

def detect_confirmation_bias(text):
    return bool(re.search(r"(only|just).*trust|ignore.*other", text))

def detect_fundamental_attribution(text):
    return bool(re.search(r"(lazy|stupid|rude).*by nature", text))

def detect_overconfidence(text):
    return bool(re.search(r"i am sure|definitely|never wrong", text))

def detect_hindsight(text):
    return bool(re.search(r"i knew it|it was obvious", text))

# =========================
# LOAD BERT MODEL
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_tokenizer = BertTokenizer.from_pretrained("bias_model")
bert_model = BertForSequenceClassification.from_pretrained("bias_model")
bert_model.to(DEVICE)
bert_model.eval()

def bert_detect(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = bert_model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    pred_id = int(torch.argmax(probs, dim=1).item())
    confidence = probs[0][pred_id].item()

    label = bert_model.config.id2label[pred_id]
    return label, confidence

# =========================
# ANALYZE ROUTE
# =========================
@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    clean = preprocess(data.text)

    if detect_sunk_cost(clean):
        return {"bias":"Sunk Cost Fallacy","severity":4,"explanation":"Past effort is influencing decision.","correction":"Evaluate future benefit only."}

    if detect_overgeneralization(clean):
        return {"bias":"Overgeneralization Bias","severity":4,"explanation":"Single event generalized broadly.","correction":"One case does not define all outcomes."}

    if detect_bandwagon(clean):
        return {"bias":"Bandwagon Effect","severity":3,"explanation":"Decision influenced by others.","correction":"Think independently."}

    if detect_confirmation_bias(clean):
        return {"bias":"Confirmation Bias","severity":3,"explanation":"Ignoring opposing views.","correction":"Seek contrary evidence."}

    if detect_fundamental_attribution(clean):
        return {"bias":"Fundamental Attribution Error","severity":3,"explanation":"Blaming personality not situation.","correction":"Consider context."}

    if detect_overconfidence(clean):
        return {"bias":"Overconfidence Bias","severity":3,"explanation":"Overestimating correctness.","correction":"Re-evaluate assumptions."}

    if detect_hindsight(clean):
        return {"bias":"Hindsight Bias","severity":2,"explanation":"Believing outcome was predictable.","correction":"Recognize uncertainty."}

    bert_bias, bert_conf = bert_detect(data.text)
    if bert_conf > 0.40:
        return {"bias":bert_bias,"severity":4 if bert_conf>0.7 else 3,"confidence":round(bert_conf,2),
                "explanation":"Bias detected using BERT contextual reasoning.",
                "correction":"Reconsider with alternative evidence."}

    return {"bias":"Possible Cognitive Bias","severity":2,
            "explanation":"System is unsure but reasoning may be distorted.",
            "correction":"Try giving an opposite example."}

@app.get("/")
def home():
    return {"status":"Bias AI backend running successfully"}
