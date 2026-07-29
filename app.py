import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    words = word_tokenize(text)

    cleaned_words = []

    for word in words:

        if word not in stop_words:

            cleaned_words.append(
                stemmer.stem(word)
            )

    return " ".join(cleaned_words)


st.title("🛒 Amazon Review Sentiment Analysis")

st.write(
    "Enter an Amazon review and let AI predict whether it is Positive or Negative."
)

review = st.text_area("Enter Review")

if st.button("Predict"):

    cleaned = clean_text(review)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    if prediction[0] == 1:

        st.success("😊 Positive Review")

    else:

        st.error("😞 Negative Review")