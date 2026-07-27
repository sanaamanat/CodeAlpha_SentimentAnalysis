import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

reviews = []

with open("dataset/train.ft.txt", "r", encoding="utf-8") as file:
    for i, line in enumerate(file):

        if i == 10000:
            break

        line = line.strip()

        sentiment, review = line.split(" ", 1)

        reviews.append([sentiment, review])

# Create DataFrame
df = pd.DataFrame(reviews, columns=["Sentiment", "Review"])

print("Dataset Loaded Successfully!\n")
print(df.head())

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Reviews:")
print(df.duplicated().sum())

# Convert labels to readable text
df["Sentiment"] = df["Sentiment"].replace({
    "__label__1": "Negative",
    "__label__2": "Positive"
})

print("\nUpdated Dataset:")
print(df.head())

print("\nSentiment Count:")
print(df["Sentiment"].value_counts())

# Create visualization
plt.figure(figsize=(6, 4))

sns.countplot(data=df, x="Sentiment")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.savefig("images/sentiment_distribution.png")
plt.show()