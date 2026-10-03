# ===============================
# TOXIC COMMENT CLASSIFICATION
# ===============================

import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (run once)
nltk.download('wordnet')
nltk.download('omw-1.4')

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("toxic_comments.csv")

print("Columns in dataset:", df.columns)
print("\nFirst 5 rows:")
print(df.head())

# -------------------------------
# Text Cleaning Function
# -------------------------------
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# Apply cleaning
df['clean_comment'] = df['comment_text'].apply(clean_text)

print("\nText cleaning completed!")

# -------------------------------
# Feature Extraction
# -------------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['toxic']

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model Training
# -------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# -------------------------------
# Model Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_comment(comment):
    cleaned = clean_text(comment)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)
    return "Toxic" if prediction[0] == 1 else "Non-Toxic"

# -------------------------------
# Test Predictions
# -------------------------------
print("\n----- TOXIC COMMENT ANALYZER -----")
user_comment = input("Enter a comment to analyze: ")

result = predict_comment(user_comment)

print("\nPrediction Result:", result)