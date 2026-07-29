# Amazon Review Sentiment Analysis

## Project Overview

This project was developed as part of the **CodeAlpha Data Analytics Internship**.

The objective is to classify Amazon customer reviews as **Positive** or **Negative** using Natural Language Processing (NLP) and Machine Learning.

The project includes:

- Exploratory Data Analysis (EDA)
- Data Visualization
- Text Preprocessing
- TF-IDF Feature Extraction
- Logistic Regression Model
- Model Evaluation
- Streamlit Web Application

---

## Dataset

Amazon Reviews Dataset

Source:
https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

The dataset contains millions of customer reviews.

For this project, the first **10,000 reviews** were used.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Streamlit
- Joblib

---

## Project Structure

```
CodeAlpha_SentimentAnalysis

│
├── app.py
├── data_analysis.py
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│
├── images/
│
└── venv/
```

---

## Project Workflow

1. Load Dataset

2. Perform Exploratory Data Analysis

3. Clean Review Text

4. Remove Stopwords

5. Apply Stemming

6. Convert Text into TF-IDF Features

7. Train Logistic Regression Model

8. Evaluate Model

9. Predict Sentiment

10. Build Streamlit Web App

---

## Results

The model predicts whether a review is:

- Positive 😊
- Negative 😞

Performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Sample Prediction

Input:

This phone is amazing.

Output:

Positive 😊

---

## Run the Project

Clone the repository

git clone https://github.com/your_username/CodeAlpha_SentimentAnalysis.git

Install dependencies

pip install -r requirements.txt

Run the analysis

python data_analysis.py

Run the web application

streamlit run app.py

## Author

Sana Amanat

CodeAlpha Data Analytics Internship
