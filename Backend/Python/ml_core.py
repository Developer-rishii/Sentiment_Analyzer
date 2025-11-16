import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from scipy.special import softmax

# -----------------------------
# Load Summarizer (LED)
# -----------------------------
model_name_sum = "allenai/led-base-16384"
tokenizer_summ = AutoTokenizer.from_pretrained(model_name_sum)
model_summ = AutoModelForSeq2SeqLM.from_pretrained(model_name_sum)

def summarize_text(text, max_len=200, min_len=50):
    inputs = tokenizer_summ(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=16384
    )

    global_attention_mask = torch.zeros_like(inputs["attention_mask"])
    global_attention_mask[:, 0] = 1

    summary_ids = model_summ.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        global_attention_mask=global_attention_mask,
        max_length=max_len,
        min_length=min_len,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer_summ.decode(summary_ids[0], skip_special_tokens=True)


# -----------------------------
# Load Sentiment Model
# -----------------------------
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
labels = ["Negative", "Neutral", "Positive"]


def perform_sentiment_analysis(text):
    text = str(text).strip()
    if not text:
        return "Neutral", [0.33, 0.34, 0.33], 0.34

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    if inputs["input_ids"].shape[1] == 0:
        return "Neutral", [0.33, 0.34, 0.33], 0.34

    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()[0]
    probs = softmax(logits)
    predicted_label = np.argmax(probs).item()
    confidence = float(np.max(probs))
    return labels[predicted_label], probs.tolist(), confidence


# -----------------------------
# CSV Processing
# -----------------------------
def process_csv(df):
    reviews = []
    sentiments = []
    confidences = []

    df = df.dropna(subset=['comment'])
    df['comment'] = df['comment'].astype(str).str.strip()
    df = df[df['comment'] != ""]

    for comment in df['comment']:
        sentiment, probs, conf = perform_sentiment_analysis(comment)
        reviews.append(comment)
        sentiments.append(sentiment)
        confidences.append(conf)

    return reviews, sentiments, confidences
