import json

from flask import Flask, jsonify, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine

import joblib
import pandas as pd
import plotly

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("./models/classifier.pkl")


# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    print(df)
    genre_counts = df.groupby('genre').count()['message']
    print('genre_counts', genre_counts)
    genre_names = list(genre_counts.index)
    print('genre_names', genre_names)
    agg_df = df.sum()[3:].sort_values()

    # categories correlation
    categories = df.iloc[:, 4:]
    category_names = list(categories.columns)
    categories_corr = categories.corr().values

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        # Distribution of Message Genres
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ),
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre",
                    'automargin': True
                }
            }
        },

        # Distribution of Message Categories
        {
            'data': [
                Bar(
                    x=agg_df.index,
                    y=agg_df.values
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle': -45,
                    'automargin': True
                }
            }
        },

        # Category Correlation Heatmap
        {
            'data': [
                Heatmap(
                    x=category_names,
                    y=category_names,
                    z=categories_corr
                )
            ],

            'layout': {
                'title': 'Correlation Heatmap of Categories',
                'xaxis': {'tickangle': -45, 'automargin': True},
                'yaxis': {'automargin': True}
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    print(ids)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
