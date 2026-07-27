import pandas as pd

reviews = []

with open("dataset/train.ft.txt", "r", encoding="utf-8") as file:

    for i, line in enumerate(file):

        if i == 10000:
            break

        line = line.strip()

        sentiment, review = line.split(" ", 1)

        reviews.append([sentiment, review])

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