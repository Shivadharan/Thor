#rule based approach(no use of nlms)
positive_words = {
    "good", "great", "excellent", "happy", "love", "amazing",
    "awesome", "fantastic", "nice", "wonderful"
}

negative_words = {
    "bad", "terrible", "awful", "sad", "hate", "poor",
    "worst", "horrible", "disappointing", "angry"
}


def predict(text):
    words = text.lower().split()

    positive_count = 0
    negative_count = 0

    for word in words:
        if word in positive_words:
            positive_count += 1
        elif word in negative_words:
            negative_count += 1

    if positive_count > negative_count:
        sentiment = "POSITIVE"
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
    else:
        sentiment = "NEUTRAL"
    total = positive_count + negative_count
    if total > 0:
        positive_percent = (positive_count / total) * 100
        negative_percent = (negative_count / total) * 100
    else:
        positive_percent = negative_percent = 0


    return sentiment, positive_count, negative_count , positive_percent , negative_percent



text = "Absolutely terrible experience, nothing worked and I regret buying it 😡💀"
sentiment, pos, neg ,pospercent , negpercent= predict(text)

print("SENTIMENT: ", sentiment)
print("POSITIVE WORD COUNT: ", pos)
print("NEGATIVE WORD COUNT: ", neg)
print("POSITIVE SCORE: " , pospercent)
print("NEGATIVE SCORE: ",negpercent)

