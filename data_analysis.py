import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()

# =====================================================
# Function: Clean Review Text
# =====================================================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            cleaned_words.append(
                stemmer.stem(word)
            )

    # Join words back into a sentence
    return " ".join(cleaned_words)



# =====================================
# Feature Extraction using TF-IDF
# =====================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(df["Clean_Review"])


y = df["Sentiment"].map({
    "Negative": 0,
    "Positive": 1
})

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)

print("Model trained successfully!")


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(f"{accuracy * 100:.2f}%")



print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:\n")
print(cm)
# =====================================================
# Load Dataset
# =====================================================

reviews = []

with open("dataset/train.ft.txt", "r", encoding="utf-8") as file:

    for i, line in enumerate(file):

        if i == 10000:
            break

        line = line.strip()

        sentiment, review = line.split(" ", 1)

        reviews.append([sentiment, review])


# =====================================================
# Create DataFrame
# =====================================================

df = pd.DataFrame(reviews, columns=["Sentiment", "Review"])

print("=" * 60)
print("Dataset Loaded Successfully!")
print("=" * 60)

print(df.head())


# =====================================================
# Basic Dataset Information
# =====================================================

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nDataset Information")
df.info()

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Reviews")
print(df.duplicated().sum())


# =====================================================
# Convert Labels
# =====================================================

df["Sentiment"] = df["Sentiment"].replace({
    "__label__1": "Negative",
    "__label__2": "Positive"
})

print("\nUpdated Dataset")
print(df.head())


# =====================================================
# Sentiment Distribution
# =====================================================

print("\nSentiment Count")
print(df["Sentiment"].value_counts())

plt.figure(figsize=(6,4))

sns.countplot(data=df, x="Sentiment")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("images/sentiment_distribution.png")

plt.show()


# =====================================================
# Review Length Analysis
# =====================================================

df["Review_Length"] = df["Review"].apply(len)

print("\nReview Length")
print(df["Review_Length"].head())

print("\nReview Length Statistics")

print(df["Review_Length"].describe())


# Histogram

plt.figure(figsize=(8,5))

plt.hist(df["Review_Length"], bins=30)

plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("images/review_length_distribution.png")

plt.show()


# Boxplot

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="Sentiment",
    y="Review_Length"
)

plt.title("Review Length by Sentiment")

plt.tight_layout()

plt.savefig("images/review_length_boxplot.png")

plt.show()


# =====================================================
# Text Preprocessing
# =====================================================

df["Clean_Review"] = df["Review"].apply(clean_text)

print("\nOriginal Review:")
print(df["Review"].iloc[0])

print("\nClean Review:")
print(df["Clean_Review"].iloc[0])


# =====================================================
# Save Clean Dataset
# =====================================================

df.to_csv("dataset/cleaned_reviews.csv", index=False)

print("\nCleaned dataset saved successfully!")