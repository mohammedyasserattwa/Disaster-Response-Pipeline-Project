import json
import pandas as pd
import plotly
from flask import Flask, render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine, inspect
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Initialize the Flask app
app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize input text.

    Parameters:
    text: str, input text

    Returns:
    clean_tokens: list of str, cleaned tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# Check if the SQLite database file and table exist
database_file = '../data/DisasterResponse.db'
table_name = 'disaster_messages'

if not os.path.exists(database_file):
    raise FileNotFoundError(f"The database file '{database_file}' does not exist.")
else:
    engine = create_engine(f'sqlite:///{database_file}')

    inspector = inspect(engine)

    if table_name not in inspector.get_table_names():
        print(inspector.get_table_names())
        raise Exception(f"The table '{table_name}' does not exist in the database.")

    # Load data from the SQLite database
    df = pd.read_sql_query(f'SELECT * FROM {table_name}', engine)

# Load the trained model
model = joblib.load("..\\models\\classifier.pkl")

# Define the index route for the web app
@app.route('/')
@app.route('/index')
def index():
    """
    Renders the index webpage with visualizations.
    """
    # Extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df1 = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_counts = df1.sum(axis=0)
    category_names = df1.columns

    # Create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # Encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # Render the web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# Define the route that handles user queries and displays model results
@app.route('/go')
def go():
    """
    Handles user query and displays model results.
    """
    # Save user input in the 'query' variable
    query = request.args.get('query', '')

    # Use the model to predict classification for the query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # Render the 'go.html' template
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

# Main function to run the Flask app
def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()
