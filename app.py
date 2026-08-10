import startup  # must be first

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from bias_detector import analyze_bias

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


class AnalyzeRequest(BaseModel):
    text: str


@app.post("/analyze")
def analyze(data: AnalyzeRequest):
    return analyze_bias(data.text)


@app.get("/")
def home():
    return {"status": "Bias AI backend running successfully", "model": "fine-tuned BERT (9 classes) + rule fallback"}