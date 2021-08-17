import pickle
import re
import sys

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection._search import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import AdaBoostClassifier

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sqlalchemy import create_engine

import pandas as pd
import numpy as np

import joblib
import nltk

# Uncomment these to download necessary packages
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

stop_words = stopwords.words()


def load_data(database_filepath: str):
    """
    Loads data from the database file and splits it into X and y
    :param database_filepath: path to the database file
    :return: df, category_names
    """
    # create an SQLight engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    # load the data
    df = pd.read_sql_table('messages', engine)

    # drop rows with incorrect data
    df = df[df.related != 2]

    # get category names
    category_names = list(df.columns[4:])

    return df, category_names


def set_sample_ratio(x, counts):
    avg = int(counts['full_count'].loc['avg'])
    x = int(x)

    if x >= avg:
        return 1
    else:
        return int(np.round(avg / x))


def get_sample_ratio(row, columns, counts):
    ratio = 1

    for i in range(4, len(row)):
        if row[i]:
            r = counts.calc_oversampling_ratio.loc[columns[i]]

            if r > ratio:
                ratio = r

    return ratio


def oversample(df):
    # dataframe of labels
    labels = df.iloc[:, 4:]
    # compute how many messages are in each category
    labels_sum = labels.sum()

    # compute average count of messages per category
    avg_val = np.mean(labels.sum())
    avg_ser = pd.Series([avg_val], index=['avg'])

    # df of counts
    counts = labels_sum.append(avg_ser, ignore_index=False).sort_values(
        ascending=False).to_frame()
    counts.columns = ['full_count']
    counts['calc_oversampling_ratio'] = counts['full_count'].apply(
        set_sample_ratio, counts=counts)

    rows = df.values.tolist()

    columns = df.columns.tolist()
    oversampled_rows = [row for row in rows for _ in range(
        get_sample_ratio(row, columns, counts))]
    df_oversampled = pd.DataFrame(oversampled_rows, columns=df.columns)

    return df_oversampled


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
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))

        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    parameters = {'classifier__estimator__learning_rate': [0.01, 0.05],
                  'classifier__estimator__n_estimators': [20, 50]}

    cv = GridSearchCV(
        pipeline,
        param_grid=parameters,
        scoring='roc_auc',
        n_jobs=-1
    )

    return cv


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

    print(classification_report(Y_test, Y_pred, target_names=category_names))


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
        df, category_names = load_data(database_filepath)

        # train test split
        msk = np.random.rand(len(df)) < 0.8
        train = df[msk]
        test = df[~msk]

        # oversample
        print('Oversampling...')
        df_oversampled = oversample(train)

        # X & Y split
        X_train = df_oversampled.message.to_numpy()
        Y_train = df_oversampled.iloc[:, 4:].values

        X_test = test.message.to_numpy()
        Y_test = test.iloc[:, 4:].values

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        # print('Loading model...')
        # model = joblib.load("models/classifier.pkl")

        print('Evaluating model...')
        try:
            evaluate_model(model, X_test, Y_test, category_names)
        except Exception as e:
            print('evaluation failed', e)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        print('Trained model saved!')
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'model/train_classifier.py data/DisasterResponse.db model/classifier.pkl')


if __name__ == '__main__':
    main()
