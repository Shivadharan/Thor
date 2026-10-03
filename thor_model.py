# thor_model.py
import pandas as pd
import emoji
import re
import ftfy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os


def clean_text(text):
    text = str(text)
    text = ftfy.fix_text(text)
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z_: ]", " ", text)
    return text.lower().strip()


def train_model():
    df = pd.read_csv("twitter_trainingclean2.csv")
    df["text"] = df["text"].apply(clean_text)
    df = df[df["labels"] != "Irrelevant"]

    texts = df["text"].tolist()
    labels = df["labels"].tolist()

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("lr", LogisticRegression(max_iter=1000))
    ])

    model.fit(texts, labels)

    # Save model
    joblib.dump(model, "thor_sentiment_model.pkl")
    return model

def load_model():
    if os.path.exists("thor_sentiment_model.pkl"):
        model = joblib.load("thor_sentiment_model.pkl")
        return model
    else:
        return train_model()

def predict_sentiment(text, model):
    cleaned = clean_text(text)
    probs = model.predict_proba([cleaned])[0]
    classes = model.classes_
    predicted = classes[probs.argmax()]
    return predicted, dict(zip(classes, probs))


load_model()
