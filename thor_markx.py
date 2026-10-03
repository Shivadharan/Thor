import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text
import os

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "text_classifier/bert_classifier/float32/1/bert_classifier.tflite"
)
MODEL_PATH = "sentiment_model.tflite"


base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = text.TextClassifierOptions(
    base_options=base_options,
    max_results=2
)

classifier = text.TextClassifier.create_from_options(options)


def analyze_sentiment(input_text: str):
    result = classifier.classify(input_text)

    positivity = 0.0
    negativity = 0.0

    for category in result.classifications[0].categories:
        label = category.category_name.lower()
        score = category.score * 100

        if "positive" in label:
            positivity = score
        elif "negative" in label:
            negativity = score

    return {
        "text": input_text,
        "positivity_percent": round(positivity, 2),
        "negativity_percent": round(negativity, 2)
    }


text_input = "I absolutely love this product, it works perfectly!"
output = analyze_sentiment(text_input)

print("INPUT:", output["text"])
print("POSITIVE:", output["positivity_percent"], "%")
print("NEGATIVE:", output["negativity_percent"], "%")
