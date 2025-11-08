import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import time
from pathlib import Path
from typing import Dict, Union, Any, List, Optional

def setup_nltk() :
    """
    Ensures the necessary NLTK resources ('punkt' for tokenization 
    and 'stopwords' for cleaning) are available on the system. 
    Downloads them quietly if they are not found.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
setup_nltk()

model_path = Path('models/spam_model.pkl')
vectorizer_path = Path('models/tfidf_vectorizer.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Models not found. The /predict endpoint will not work.")
    model = None
    vectorizer = None
except Exception as e:
    print("Error loading model or vectorizer.")
    model = None
    vectorizer = None

data = pd.read_csv('data/spam.csv', encoding='latin-1')
data = data.rename(columns={'v1': 'label', 'v2': 'message'})
# Statistics of the dataset
dataset_stats = {
        "shape": data.shape,
        "label_counts": data['label'].value_counts().to_dict(),
        "label_proportions": (data['label'].value_counts(normalize=True) * 100).round(2).to_dict(),
        "columns": data.columns.tolist(),
        "missing_values": data.isnull().sum().to_dict()
    }

# Step 2
# Text Preprocessing Function
def text_prepro(text) :
    text = text.lower() # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Keeping only letters and spaces
    tokens = word_tokenize(text)  # Tokenizing (splitting into words)
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(cleaned_tokens)

# FastAPI Set up
app = FastAPI(title="Spam Detection API")
class TextPayLoad(BaseModel):
    text:str

@app.get("/", tags=["Basic"])
# Returns basic information about the API.
def read_root():
    return {
        "project": "Spam Detection API",
        "version": "1.0",
        "endpoints": ["/dataset-info", "/preprocess"]
    }

@app.get("/dataset-info", tags=["Analysis"])
def get_dataset_info():
    return dataset_stats

@app.post("/preprocess", tags=["Text Processing"])
def preprocess_text_endpointmessage (payload: TextPayLoad):
    """Accepts a single raw message and processes it using the text preprocessing function"""
    raw_text = payload.text
    cleaned_text = text_prepro(raw_text)
    return {
        "Original_text" : raw_text,
        "cleaned_text" : cleaned_text
    }

@app.post("/predict", tags=["Prediction"])
def predict_spam(payload: TextPayLoad):
    """Accepts a single raw text message, processes it using the trained 
    TF-IDF vectorizer and Logistic Regression model, and returns a 
    spam/ham prediction with the associated confidence score."""
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Cannot make prediction."}
    start_time = time.time()
    raw_message = payload.text
    cleaned_message = text_prepro(raw_message) # Preprocessing
    message_vector = vectorizer.transform([cleaned_message]) # Vectorization
    prediction = model.predict(message_vector)[0]
    probabilities = model.predict_proba(message_vector)[0] # Confidence Score
    if prediction == 'spam':
        confidence = probabilities[model.classes_ == 'spam'][0]
    else:
        confidence = probabilities[model.classes_ == 'ham'][0]
    end_time = time.time()
    return {"prediction": prediction, "confidence_score": round(float(confidence), 4), "raw_message": raw_message, "processing_time_s": round(end_time - start_time, 4)}
