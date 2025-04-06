import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import joblib

# Download NLTK stopwords (run only once)
nltk.download('stopwords')

# Load training data
train_path = r"C:\Users\Shrivatsa\Downloads\training_set_rel3 (2).tsv"  # Use raw string to handle backslashes
df = pd.read_csv(train_path, sep='\t', usecols=['essay', 'domain1_score'], encoding='ISO-8859-1')


# Text Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    return " ".join([word.lower() for word in text.split() if word.lower() not in stop_words])

df['processed_essay'] = df['essay'].apply(preprocess_text)

# Create a Pipeline: TF-IDF Vectorization + Ridge Regression Model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('ridge', Ridge(alpha=1.0))
])

# Train Model
pipeline.fit(df['processed_essay'], df['domain1_score'])

# Save Model
joblib.dump(pipeline, "essay_grader.pkl")
print("Model training completed and saved as 'essay_grader.pkl'.")
