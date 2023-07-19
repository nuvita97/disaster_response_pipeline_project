import sys
import pandas as pd
import numpy as np
import nltk
import pickle
from nltk import punkt, wordnet
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    """ Load data from SQLite database """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('messages', engine)

    X = df['message'].values
    Y = df.iloc[:, 4:].values

    category_names = list(df.iloc[:, 4:].columns)
    return X, Y, category_names


def tokenize(text):
    """ Tokenize data """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_tok = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens



def build_model():
    """ Build model pipeline """

    pipeline = Pipeline(
        [('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))]
    )

    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate model performance """

    Y_pred = model.predict(X_test)

    index = 0
    for category in category_names:
        print("Category in column {}: {}".format(index, category))
        evaluation_report = classification_report(Y_test[:, index], Y_pred[:, index])
        index += 1
        print(evaluation_report)


def save_model(model, model_filepath):
    """ Save model as a pickle file """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()