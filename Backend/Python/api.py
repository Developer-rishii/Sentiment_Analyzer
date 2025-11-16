from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
from ml_core import perform_sentiment_analysis, summarize_text, process_csv

app = FastAPI()


# -----------------------------
# Single text sentiment
# -----------------------------
@app.post("/sentiment")
async def sentiment_endpoint(text: str = Form(...)):
    sentiment, probs, conf = perform_sentiment_analysis(text)
    return {
        "sentiment": sentiment,
        "probabilities": probs,
        "confidence": conf
    }


# -----------------------------
# File upload (CSV)
# -----------------------------
@app.post("/analyze_csv")
async def analyze_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    reviews, sentiments, confs = process_csv(df)

    return {
        "total": len(reviews),
        "reviews": reviews,
        "sentiments": sentiments,
        "confidences": confs
    }


# -----------------------------
# Large text summarization
# -----------------------------
@app.post("/summarize")
async def summarize_endpoint(text: str = Form(...)):
    summary = summarize_text(text)
    return {"summary": summary}
