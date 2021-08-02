import pickle
import re
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

import joblib
import nltk

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = stopwords.words()


def load_data(database_filepath: str):
    """
    Loads data from the database file and splits it into X and y
    :param database_filepath: path to the database file
    :return: X, y, category_names
    """
    # create an SQLight engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # load the data
    df = pd.read_sql_table('messages', engine)

    # drop rows with incorrect data
    df = df[df.related != 2]

    # split into X & y
    X = df.message.to_numpy()
    y = df.iloc[:, 4:].values

    # get category names
    category_names = list(df.columns[4:])

    return X, y, category_names


def tokenize(text: str):
    """
    Gets a text string and tokenizes it.
    :param text: string to be tokenized
    :return: list of lemmatized tokens with stopwords removed
    """
    # remove punctuation
    text_clean = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # normalize and tokenize
    tokens = word_tokenize(text_clean.lower())

    # init lemmatizer
    lemmatizer = WordNetLemmatizer()

    # remove stop words and lemmatize
    clean_tokens = []
    for token in tokens:
        clean_tokens = [
            lemmatizer.lemmatize(word.strip()) for word in tokens if word not in stop_words
        ]

    return clean_tokens


def build_model():
    """
    Creates a Random Forest model using pipeline for feature extraction
    :return: model
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """

    :param model: model to be evaluated
    :param X_test: test features
    :param Y_test: test labels
    :param category_names: list of category names
    :return: classification report
    """
    print('Predicting ...')
    Y_pred = model.predict(X_test)

    print(classification_report(
        Y_test,
        np.array([x[:] for x in Y_pred]),
        target_names=category_names)
    )


def save_model(model, model_filepath: str):
    """

    :param model: model to be saved
    :param model_filepath: path (str) where to save the model
    :return: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        # print('Loading model...')
        # model = joblib.load("classifier.pkl")

        print('Evaluating model...')
        try:
            evaluate_model(model, X_test, Y_test, category_names)
        except Exception as e:
            print('evaluation failed', e)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
