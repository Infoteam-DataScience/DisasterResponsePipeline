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


def accuracy(model, yT, yP):
    '''
    Calculate accuracy of handed in data-sets
    '''    
    labels = np.unique(yP)        
    
    print('Accuracy: ', (yT==yP).mean())


def report(dataframe, pipeline):
    '''
    Classification report for all the columns
    '''
    
    # make prediction
    Y_pred = pipeline.predict(X_test)    
    
    # list of all column names
    targets = [item for item in dataframe.drop(columns=ignore).columns]
        
    # iterate through all columns
    for index in range(0, len(targets)):    
        
        print(classification_report(
            Y_test[:, index], 
            Y_pred[:, index], 
            target_names=[targets[index]])
         )  


def tokenize(text):
    '''
    Tokenize engine for the feature generation
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    for tok in tokens:
        clean_tokens.append(lemmatizer.lemmatize(tok).lower().strip())

    return clean_tokens

def main():

    # creating engine for sql
    engine = create_engine('sqlite:///' + sys.argv[1])

    # load data from database
    df = pd.read_sql_table('InsertTableName', engine)

    # input values for model.
    X = df.message.values

    # output values for model
    ignore = ['id', 'message', 'original', 'genre']
    Y = df.drop(columns=ignore).values

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

    # split data set into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)    

    # parameters for grid search optimization
    parameters = {        
        'features__text_count__vectorizations__max_df': (0.5, 0.75, 1.0),        
        'features__text_count__transformation__use_idf': (True, False)
    }

    # specify all parameters for grid search.
    model = GridSearchCV(pipeline, param_grid=parameters)   

    # train and optimize the model parameters
    model.fit(X_train, Y_train)     

    # save the trained pipeline into pickles.
    pickle.dump(model, open(sys.argv[2], 'wb'))
   
if __name__ == "__main__":
    main()