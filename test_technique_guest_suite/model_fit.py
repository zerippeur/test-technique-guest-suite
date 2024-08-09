import time
from contextlib import contextmanager

import numpy as np
import pickle
import gzip
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data
    Function to load the train and validation data.
    """

    # Load data
    with gzip.open("../data/train_no_nan.pkl.gz", "rb") as f:
        train_df = pickle.load(f)

    with gzip.open("../data/val_no_nan.pkl.gz", "rb") as f:
        val_df = pickle.load(f)
        
    X_train = train_df["review_text"]
    X_val= val_df["review_text"]
    y_train = train_df["global_rate"]
    y_val = val_df["global_rate"]
    return X_train, X_val, y_train, y_val

def create_stopwords_set() -> set:
    """
    Create stopwords set
    Function to create a set of stopwords for all supported languages.
    """
    nltk.download('stopwords')
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

def train_and_evaluate_model(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.DataFrame,
    y_val: pd.DataFrame
) -> None:
    """
    Train and evaluate model
    Function to train and evaluate the model.
    """

    # Train model
    with timer("training model"):
        # Create the pipeline
        model = make_pipeline(TfidfVectorizer(), LogisticRegression(solver='lbfgs', max_iter=1000))

        # Train the pipeline
        results = cross_validate(model, X_train, y_train, cv=5, scoring=['accuracy', 'f1_macro'], return_train_score=False)
        print("Résultats de la validation croisée :", results)
    
    # Evaluate model
    with timer("fitting model"):
        model.fit(X_train, y_train)

    # Make predictions
    with timer("computing predictions on validation set"):
        y_pred = model.predict(X_val)

    # Evaluate the model
    with timer("evaluating model and plotting confusion matrix"):
        accuracy = accuracy_score(y_val, y_pred)
        print(f'accuracy: {np.sqrt(accuracy):.2f}')

        # Creation of the confusion matrix
        cm = confusion_matrix(y_val, y_pred, labels=['Détracteur', 'Neutre', 'Promoteur'])

        # Conversion of the confusion matrix into a DataFrame
        cm_df = pd.DataFrame(cm, index=['Détracteur', 'Neutre', 'Promoteur'], columns=['Détracteur', 'Neutre', 'Promoteur'])

        # Creation of the heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_df.values,
            x=cm_df.columns,
            y=cm_df.index,
            colorscale=['White', 'Darkturquoise'],
            text=cm_df.values,
            texttemplate="%{text}",
            textfont={"size":20},
        ))

        fig.update_layout(
            title='Matrice de Confusion',
            title_x=0.5,
            xaxis_title="Prédictions",
            yaxis_title="Vraies Valeurs"
        )

        fig.write_image("confusion_matrix.png")

        # Creation of the classification report
        report = classification_report(y_val, y_pred, output_dict=True)

        # Conversion of the classification report into a DataFrame
        report_df = pd.DataFrame(report).transpose()

        # Columns to format
        columns_to_format = ['precision', 'recall', 'f1-score']
        report_df[columns_to_format] = report_df[columns_to_format].applymap(
            lambda x: f"{x:.2f}" if isinstance(x, float) else x
        )

        # Switch index to a column
        report_df.reset_index(names='', inplace=True)

        # Correction of the wrong values
        report_df.at[3, 'support']  = report_df.at[4, 'support']

        # Creation of the table
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(report_df.columns),
                        fill_color='darkturquoise',
                        align='left'),
            cells=dict(values=[report_df[col] for col in report_df.columns],
                    fill_color='paleturquoise',
                    align='left'))
        ])

        fig.update_layout(title_text='Classification Report',
                          title_x=0.5)

        # Saving the classification report
        fig.write_image('classification_report.png')

    # Saving the model
    with timer("saving model"):
        with gzip.open("model.pkl.gz", "wb") as f:
            pickle.dump(model, f)


def main() -> None:
    """
    Main function
    Function to train and evaluate the model.
    """

    # Load data
    with timer("loading data"):
        X_train, X_val, y_train, y_val = load_data()

    # Create stopwords set
    with timer("creating stopwords set"):
        all_stopwords = create_stopwords_set()

    # Preprocess text
    with timer("preprocessing text"):
        X_train = X_train.apply(lambda x: preprocess_text(text=x, stopwords=all_stopwords))
        X_val = X_val.apply(lambda x: preprocess_text(text=x, stopwords=all_stopwords))

    # Categorize NPS
    with timer("transforming target with NPS categories"):
        y_train = y_train.apply(categorize_nps)
        y_val = y_val.apply(categorize_nps)

    # Train and evaluate model
    with timer("training and evaluating model"):
        train_and_evaluate_model(X_train, X_val, y_train, y_val)
    
main()