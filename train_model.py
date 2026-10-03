import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer
import joblib

# Download NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
df = pd.read_csv("toxic_comments.csv")  # Make sure this CSV is in the same folder

# Text cleaning
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

df['clean_comment'] = df['comment_text'].apply(clean_text)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['clean_comment'])
y = df['toxic']

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "toxic_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")