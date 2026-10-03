#implementing machine learning and naive bayes
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

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

model = Pipeline([("tfidf",TfidfVectorizer()),("nb",MultinomialNB())])

model.fit(texts,labels)

text = ["i would love to join rig"]
#predict prb must me a list
probabilty = model.predict_proba(text)[0]
order = model.classes_

print(order[0],"=", round(probabilty[0]*100 , 2))
print(order[1],"=", round(probabilty[1]*100 , 2))


