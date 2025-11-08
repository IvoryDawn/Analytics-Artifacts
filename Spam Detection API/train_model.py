import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
import re
from pathlib import Path
import pickle

def setup_nltk() :
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

def preprocess_text(text: str) -> str :
    if not isinstance(text, str) or not text.strip():
        return ""
    try:
        text = text.lower().strip() # Convert to lowercase and strip spaces
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'[^a-z\s]', ' ', text) # Removes non-letters with space
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        cleaned_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        return ' '.join(cleaned_tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""  # Return empty string on failure
    
def train_pipeline():
    start_time = time.time()
    try:
        data = pd.read_csv('Spam Detection API/data/spam.csv', encoding='latin-1')
        data = data.rename(columns = {'v1':'label', 'v2': 'message'})
        data = data[['label', 'message']]
        print("Data loaded successfully. Shape : ", data.shape)
    except FileNotFoundError:
        print("Error: 'data/spam.csv' not found. Please check the file path.")
        return
    except KeyError:
        print("Error: Required columns 'v1', or 'v2' not in CSV. Check file format.")
        return
    
    # Preprocessing data
    data['cleaned_message'] = data['message'].apply(preprocess_text)
    # Define features X and Y
    x = data['cleaned_message']
    y = data['label']

    # Split data (Train and test)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2, stratify = y)
    print("Data split: ", len(y_train), " train ", len(y_test), " test samples.")

    # TF-IDF Vectorization
    tfidf_vect = TfidfVectorizer()
    x_train_tfidf = tfidf_vect.fit_transform(x_train)
    x_test_tfidf = tfidf_vect.transform(x_test)
    print("Text Vectorized. Feature matrix shape: ", x_train_tfidf.shape)

    # Train Logistic Regression Model
    model = LogisticRegression(class_weight= 'balanced', random_state = 42, max_iter = 1000)
    model.fit(x_train_tfidf, y_train)

    # Evaluate Model
    y_pred = model.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    print("Accuracy: ", (accuracy*100))
    print("Classification Report: \n", report)

    # Save the model and Vectorizer
    print("Saving model and vectorizer to the disk...")
    model_dir = Path('Spam Detection API/models')
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir/'spam_model.pkl'
    vectorizer_path = model_dir/'tfidf_vectorizer.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(tfidf_vect, f)

    end_time = time.time()
    print("successfullly saved model to: ", model_path)
    print("Successfully saved vectorizer to: ", vectorizer_path)
    print("Pipeline finished in ", (end_time - start_time), " seconds")

setup_nltk()
train_pipeline()