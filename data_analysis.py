# ==========================================================
# CodeAlpha Internship Project
# Project: Sentiment Analysis on Amazon Reviews
# Author: Sana Amanat
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Download NLTK Resources
# ==========================

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# ==========================
# NLP Objects
# ==========================

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ==========================
# Text Cleaning Function
# ==========================

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords and apply stemming
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            cleaned_words.append(stemmer.stem(word))

    return " ".join(cleaned_words)

# ==========================
# Load Dataset
# ==========================

reviews = []

with open("dataset/train.ft.txt", "r", encoding="utf-8") as file:

    for i, line in enumerate(file):

        if i == 10000:
            break

        line = line.strip()

        sentiment, review = line.split(" ", 1)

        reviews.append([sentiment, review])

# ==========================
# Create DataFrame
# ==========================

df = pd.DataFrame(
    reviews,
    columns=["Sentiment", "Review"]
)

print("=" * 60)
print("Dataset Loaded Successfully")
print("=" * 60)

print(df.head())

# ==========================
# Basic EDA
# ==========================

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

# ==========================
# Convert Labels
# ==========================

df["Sentiment"] = df["Sentiment"].replace({
    "__label__1": "Negative",
    "__label__2": "Positive"
})

print("\nSentiment Distribution")
print(df["Sentiment"].value_counts())

# ==========================
# Visualization 1
# ==========================

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="Sentiment"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")

plt.tight_layout()

plt.savefig("images/sentiment_distribution.png")

plt.show()

# ==========================
# Review Length
# ==========================

df["Review_Length"] = df["Review"].apply(len)

print("\nReview Length Statistics")

print(df["Review_Length"].describe())

# ==========================
# Histogram
# ==========================

plt.figure(figsize=(8,5))

plt.hist(
    df["Review_Length"],
    bins=30
)

plt.title("Distribution of Review Lengths")
plt.xlabel("Review Length")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig("images/review_length_distribution.png")

plt.show()

# ==========================
# Boxplot
# ==========================

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

# ==========================
# Text Preprocessing
# ==========================

df["Clean_Review"] = df["Review"].apply(clean_text)

print("\nOriginal Review:\n")
print(df["Review"].iloc[0])

print("\nClean Review:\n")
print(df["Clean_Review"].iloc[0])

# ==========================
# Save Clean Dataset
# ==========================

df.to_csv(
    "dataset/cleaned_reviews.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")

# ==========================
# Feature Extraction
# ==========================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    df["Clean_Review"]
)

# ==========================
# Labels
# ==========================

y = df["Sentiment"].map({
    "Negative": 0,
    "Positive": 1
})

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==========================
# Logistic Regression Model
# ==========================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

print("\nModel Trained Successfully!")

# ==========================
# Prediction
# ==========================

y_pred = model.predict(
    X_test
)

# ==========================
# Accuracy
# ==========================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

# ==========================
# Classification Report
# ==========================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ==========================
# Confusion Matrix
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix\n")

print(cm)

# ==========================
# Confusion Matrix Heatmap
# ==========================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("images/confusion_matrix.png")

plt.show()

# ==========================
# Predict New Review
# ==========================

def predict_sentiment(review):

    # Clean the review
    cleaned_review = clean_text(review)

    # Convert into TF-IDF features
    review_vector = vectorizer.transform([cleaned_review])

    # Predict sentiment
    prediction = model.predict(review_vector)

    if prediction[0] == 1:
        return "Positive 😊"
    else:
        return "Negative 😞"

    print("\n" + "=" * 60)
print("Predict Sentiment for Your Own Review")
print("=" * 60)

user_review = input("\nEnter your review:\n")

result = predict_sentiment(user_review)

print("\nPrediction:", result)