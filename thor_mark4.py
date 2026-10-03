#making thor learn from custom data with logistic regression
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
#training data
df = pd.read_csv("train_sentiment.csv")
val_df = pd.read_csv("val_sentiment.csv")

texts = df["text"].tolist()
labels = df["labels"].tolist()

texts_val = df["text"].tolist()
labels_val = df["labels"].tolist()

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)
val_preds = model.predict(texts_val)
report = classification_report(labels_val, val_preds)
print(report)

probabilty = model.predict_proba(["thankyou for visiting my place but please never come again"])[0]
order = model.classes_

print(order[0],"=", round(probabilty[0]*100 , 2))
print(order[1],"=", round(probabilty[1]*100 , 2))

