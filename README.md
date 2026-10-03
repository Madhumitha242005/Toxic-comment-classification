# 🚨 Toxic Comment Classification Using Natural Language Processing

## 📌 Project Overview

Toxic Comment Classification is an NLP-based project that identifies whether a given comment is **Toxic** or **Non-Toxic**.

The system processes the input text using Natural Language Processing techniques such as text cleaning, tokenization, and lemmatization. TF-IDF is used to convert text into numerical features, and Logistic Regression is used to classify the comments.

The project also includes a simple **Streamlit web application** where users can enter one or multiple comments and get the classification result.

---

## 🎯 Purpose

The main purpose of this project is to automatically detect toxic comments in online text.

It helps to:

- Identify toxic and non-toxic comments.
- Reduce manual comment moderation.
- Process multiple comments efficiently.
- Demonstrate practical Natural Language Processing techniques.
- Provide a simple and user-friendly interface for testing comments.

---

## ✨ Features

- 📝 Enter comments through a web interface.
- 🔍 Automatically preprocess text.
- 🧹 Text cleaning and normalization.
- 🔤 Tokenization and lemmatization.
- 📊 TF-IDF feature extraction.
- 🤖 Toxic/Non-Toxic text classification.
- 📈 Displays classification confidence.
- 📋 Supports multiple comments.
- 🌐 Built using Streamlit.

---

## 🛠️ Technologies Used

- **Python**
- **Natural Language Processing (NLP)**
- **Pandas**
- **NLTK**
- **Scikit-learn**
- **TF-IDF**
- **Logistic Regression**
- **Streamlit**
- **Joblib**

---

## 🧠 NLP Process

The project follows these main steps:

### 1. Text Input

The user enters a comment through the Streamlit application.

### 2. Text Preprocessing

The input text is:

- Converted to lowercase
- Cleaned by removing unwanted characters
- Tokenized into words
- Lemmatized using NLTK

### 3. Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) converts the processed text into numerical features.

### 4. Classification

Logistic Regression classifies the processed comment into:

- **Toxic**
- **Non-Toxic**

### 5. Result

The application displays the predicted category along with the classification confidence.

---
Algorithm Used

The project uses a Supervised NLP-based Text Classification approach.

TF-IDF

TF-IDF is used for extracting important numerical features from text.

Logistic Regression

Logistic Regression is used as the classification algorithm to identify whether a comment is toxic or non-toxic.

Simple Workflow
Input Comment
      ↓
Text Preprocessing
      ↓
Tokenization
      ↓
Lemmatization
      ↓
TF-IDF Feature Extraction
      ↓
Logistic Regression
      ↓
Toxic / Non-Toxic

Sample Input
Toxic Comments
You are stupid.
Stop talking nonsense, you idiot.
Nobody wants to hear your stupid comments.
Non-Toxic Comments
Thank you for sharing your helpful idea.
I really appreciate your support.
Could you please explain your point again?
📌 Example Output
🚫 Toxic Comment
Confidence: 94.25%

or

✅ Non-Toxic Comment
Confidence: 91.63%
📈 Advantages
Fast text classification.
Simple and easy-to-use interface.
Supports multiple comments.
Reduces manual moderation effort.
Demonstrates practical NLP techniques.
Can be extended for larger datasets.


## 📂 Project Structure

```text
toxic-comment-classification/
│
├── app.py
├── toxic_comment.py
├── toxic_model.pkl
├── vectorizer.pkl
├── toxic_comments.csv
├── requirements.txt
└── README.md
