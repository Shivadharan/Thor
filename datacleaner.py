import pandas as pd

# Load CSV WITHOUT header
df = pd.read_csv("twitter_training.csv", header=None)

# Assign correct column names
df.columns = ["id", "entity", "labels", "texts"]

# Normalize labels
df["labels"] = df["labels"].str.lower()

# Keep only positive and negative
df = df[df["labels"].isin(["positive", "negative"])]

# Keep only required columns
df = df[["texts", "labels"]]

# Drop empty rows
df = df.dropna().reset_index(drop=True)

# Save clean CSV
df.to_csv("sentiment_clean_full.csv", index=False)

print("Done. File saved as sentiment_clean_full.csv")
print(df.head())
