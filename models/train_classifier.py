import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    """Loads data from sqlite database and returnes
     messages and categories dataframe as X and Y.
    """

    # creating sqllite engine
    engine = create_engine('sqlite:///'+database_filepath)
    
    # quering MessagesCategorized dataset
    df = pd.read_sql("SELECT * FROM MessagesCategorized", engine)

    # Spliting data into messages and categories dataframes 
    X = df["message"]
    Y = df.iloc[:, 4:]

    return X, Y


def tokenize(message):
    """Tokenization function which performs normalization, 
    tokenization and lemmatization of messages.
    """

    # normalizing messages
    message = re.sub(r"[^a-zA-Z0-9]", " ", message.lower())

    # tokenization
    tokens = word_tokenize(message)

    # lemmatization
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Creating machine learning model for classification messages on the 36 categories"""
    
    # creating pipeline 
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def display_results(y_test, y_pred):
    """Function for displaying results Rof the f1 score, precision 
    and recall for each output category of the dataset"""
    
    # creating metrics
    class_report = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    # displaying metrics
    print("Classification Report:\n", class_report)
    print("Accuracy:", accuracy)


def evaluate_model(model, X_test, Y_test):
    """Function for evaluating model"""

    # predicting results on test data
    Y_pred = model.predict(X_test)
    
    # displaying results and estimating averaged accuracy
    accuracy_sum = 0
    for column in Y_test:
        Y_test_one_col = Y_test[column]  
        Y_pred_one_col = Y_pred[:,Y_test.columns.get_loc(column)]
        print("Category: ", column)
        display_results(Y_test_one_col, Y_pred_one_col)
        print("\n")
        accuracy_sum+=accuracy_score(Y_test_one_col, Y_pred_one_col)

    # displaying averaged accuracy 
    accuracy_avg = accuracy_sum/Y_test.shape[1]
    print("Accuracy averaged: ", accuracy_avg)


def improve_model(model, X_train, Y_train):
    """Function for improving model by performing grid search"""
    
    # defining parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3]
    }   

    # initializing grid search 
    cv = GridSearchCV(model, param_grid=parameters)

    # performing grid search
    cv.fit(X_train, Y_train)

    # printing best parameters
    print("\nBest Parameters:", cv.best_params_)

    return cv


def save_model(model, model_filepath):
    """Function for saving model into pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath) 
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test) 

        print('Improving model with Grid Search...')
        model = improve_model(model, X_train, Y_train) # this part is optional, grid search takes a lot of time to process

        print('Evaluating improved model...')
        evaluate_model(model, X_test, Y_test) 

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