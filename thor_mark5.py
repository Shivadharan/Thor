#making thor to learn from real life data
import pandas as pd
import emoji
import re
import ftfy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def clean_text(text):
    text = str(text)
    text = ftfy.fix_text(text)
    text = emoji.demojize(text)            
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z_: ]", " ", text)
    return text.lower().strip()




#training data
df = pd.read_csv("twitter_trainingclean2.csv")
df["text"] = df["text"].apply(clean_text)
val_df = pd.read_csv("twitter_validationclean2.csv")
val_df["text"] = val_df["text"].apply(clean_text)

df = df[df["labels"] != "Irrelevant"]
val_df = val_df[val_df["labels"] != "Irrelevant"]


texts = df["text"].tolist()
labels = df["labels"].tolist()

texts_val = val_df["text"].tolist()
labels_val = val_df["labels"].tolist()

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)
val_preds = model.predict(texts_val)
report = classification_report(labels_val, val_preds)
print(report)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(labels_val, val_preds)
print(f"Accuracy: {accuracy * 100:.2f}%")


probabilty = model.predict_proba(["I absolutely love this product, it works perfectly!"])[0]
order = model.classes_

print(order[0],"=", round(probabilty[0]*100 , 2))
print(order[1],"=", round(probabilty[1]*100 , 2))
print(order[2],"=", round(probabilty[2]*100 , 2))

