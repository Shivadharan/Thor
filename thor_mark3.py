#implementing machine learing using logistic regression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
#training data
texts = [
    "I love this movie",
    "This product is amazing",
    "Very happy with the service",
    "I hate this item",
    "This is the worst experience",
    "Terrible quality and bad support"
]

labels = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative"
]
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("lr", LogisticRegression(max_iter=1000))
])

model.fit(texts, labels)

probabilty = model.predict_proba(["rig is challenging but i love it"])[0]
order = model.classes_

print(order[0],"=", round(probabilty[0]*100 , 2))
print(order[1],"=", round(probabilty[1]*100 , 2))

