import time
from contextlib import contextmanager

import pickle
import gzip
import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import nltk
import re

@contextmanager
def timer(title: str):
    """
    A context manager that measures the time taken to execute a block of code.

    Args:
        title (str): The title of the code block.

    Yields:
        None

    Prints:
        The title of the code block and the time taken to execute it.

    Example:
        >>> with timer("Function execution"):
        ...     # code block
        Function execution - done in 5s
    """
    start_time = time.time()
    yield
    print(f"{title} - done in {time.time() - start_time:.0f}s")

def load_data(
) -> pd.DataFrame:
    """
    Load data
    Function to load the test data.
    """

    # Load data
    with gzip.open("./data/test_no_nan.pkl.gz", "rb") as f:
        test_df = pickle.load(f)
        
    return test_df

def load_model() -> Pipeline:
    """
    Load model
    Function to load the trained model.
    """
    with gzip.open("./model.pkl.gz", "rb") as f:
        model = pickle.load(f)
    return model

def create_stopwords_set() -> set:
    """
    Create stopwords set
    Function to create a set of stopwords for all supported languages.
    """
    nltk.download('stopwords', download_dir='./data')
    # Create a combined list of stopwords for all supported languages
    all_stopwords = set()
    for language in stopwords.fileids():
        all_stopwords.update(stopwords.words(language))
    return all_stopwords

def preprocess_text(text: str, stopwords: set) -> str:
    """
    Preprocess text
    Function to preprocess the text by removing special characters and digits, except for Chinese/Japanese characters.
    """
    # Remove non-alphabetic characters, except for accented characters and Chinese/Japanese characters
    text = re.sub(r'[^a-zA-Z0-9\sà-ÿÀ-Ÿ\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    # Remove extra whitespace
    text = text.strip()
    return text

# Fonction pour catégoriser les notes selon le NPS
def categorize_nps(score: float) -> str:
    if score >= 9:
        return 'Promoteur'
    elif score >= 7:
        return 'Neutre'
    else:
        return 'Détracteur'

def predict(model: Pipeline, X_test: pd.DataFrame) -> pd.DataFrame:
    """
    Predict
    Function to predict the target variable using the trained model.
    """

    y_pred = model.predict(X_test)
    y_pred = [categorize_nps(score) for score in y_pred]

def main() -> None:
    """
    Main function
    Function to train and evaluate the model.
    """

    # Load data
    with timer("loading data"):
        test_df = load_data()

    # Create stopwords set
    with timer("creating stopwords set"):
        all_stopwords = create_stopwords_set()

    # Preprocess text
    with timer("preprocessing text"):
        test_df["review_text"] = test_df["review_text"].apply(lambda x: preprocess_text(text=x, stopwords=all_stopwords))

    # Load model
    with timer("loading model"):
        model = load_model()

    # Predict
    with timer("predicting"):
        y_pred = model.predict(test_df["review_text"])
        test_df["prediction"] = y_pred

    # Save predictions
    with timer("saving predictions"):
        test_df.to_csv("./data/predictions.csv", index=False)
    
if __name__ == "__main__":
  main()