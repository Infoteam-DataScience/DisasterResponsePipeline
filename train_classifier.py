# import libraries
import sys

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pickle

import numpy as np

import pandas as pd

from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):

    # creating engine for sql
    engine = create_engine('sqlite:///' + database_filepath)

    # load data from database
    df = pd.read_sql_table('InsertTableName', engine)

    # input values for model.
    X = df.message.values

    # output values for model
    ignore = ['id', 'message', 'original', 'genre']
    Y = df.drop(columns=ignore).values

    # list of all column names
    category_names = [item for item in dataframe.drop(columns=ignore).columns]

    return X, Y, category_names


def tokenize(text):
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())

    return clean_tokens


def build_model():

    # creates new ml pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_count', Pipeline([
                ('vectorizations', CountVectorizer(tokenizer=tokenize)),
                ('transformation', TfidfTransformer())
            ])),

            ('text_hashs', Pipeline([
                ('HashingMethods', HashingVectorizer(tokenizer=tokenize)),
                ('transformation', TfidfTransformer())
            ])),            
            
        ])),

        ('classification', MultiOutputClassifier(
            VotingClassifier(
            
                estimators=[
                    ('randomforest', RandomForestClassifier()),
                    ('gradboosting', GradientBoostingClassifier())
                ]

        )))
    ])

    # parameters for grid search optimization
    parameters = {        
        'features__text_count__vectorizations__max_df': (0.5, 0.75, 1.0),        
        'features__text_count__transformation__use_idf': (True, False)
    }

    # specify all parameters for grid search.
    model = GridSearchCV(pipeline, param_grid=parameters)   

    return model


def evaluate_model(model, X_test, Y_test, category_names):
     
    # make prediction
    Y_pred = model.predict(X_test)    
        
    # iterate through all columns
    for index in range(0, len(category_names)):    
        
        print(classification_report(
            Y_test[:, index], 
            Y_pred[:, index], 
            target_names=[category_names[index]])
         )  


def save_model(model, model_filepath):
    
    # save the trained pipeline into pickles.
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