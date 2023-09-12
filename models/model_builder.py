import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import classification_report

nltk.download(['punkt', 'wordnet'])

def tokenize(text):
    """
    Tokenizes and lemmatizes text.

    Parameters:
    text: Text to be tokenized

    Returns:
    clean_tokens: Returns cleaned tokens
    """
    # Tokenize text
    tokens = word_tokenize(text)

    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Iterate through each token
    clean_tokens = []
    for tok in tokens:
        # Lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.

    Returns:
    cv: Classifier
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators': [50, 100]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of the model and returns a classification report.

    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test

    Returns:
    Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

def save_model(model, model_filepath):
    """Exports the final model as a pickle file."""
    joblib.dump(model, model_filepath)
