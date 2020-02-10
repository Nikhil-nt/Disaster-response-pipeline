import nltk
import sys
nltk.download('punkt')
nltk.download('wordnet')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3 
import nltk
import re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib 

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql('Select * from dataframe1',engine)
    X= df['message']
    Y= df[['aid_centers', 'aid_related', 'buildings', 'child_alone', 'clothing',
       'cold', 'death', 'direct_report', 'earthquake', 'electricity', 'fire',
       'floods', 'food', 'hospitals', 'infrastructure_related',
       'medical_help', 'medical_products', 'military',
       'missing_people', 'money', 'offer', 'other_aid',
       'other_infrastructure', 'other_weather', 'refugees', 'related',
       'request', 'search_and_rescue', 'security', 'shelter', 'shops', 'storm',
       'tools', 'transport', 'water', 'weather_related']].values
    Z= df[['aid_centers', 'aid_related', 'buildings', 'child_alone', 'clothing',
       'cold', 'death', 'direct_report', 'earthquake', 'electricity', 'fire',
       'floods', 'food', 'hospitals', 'infrastructure_related',
       'medical_help', 'medical_products', 'military',
       'missing_people', 'money', 'offer', 'other_aid',
       'other_infrastructure', 'other_weather', 'refugees', 'related',
       'request', 'search_and_rescue', 'security', 'shelter', 'shops', 'storm',
       'tools', 'transport', 'water', 'weather_related']]
    return X,Y,Z
    pass

def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
    pass




def build_model():
    vect = CountVectorizer()
    tfidf = TfidfTransformer()
    clf = MultiOutputClassifier(vect)

    model = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    return model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    # Try removing this
    for columns in category_names:
        print(classification_report(Y_test,y_pred))

    
    accuracy = (y_pred == Y_test).mean()
    pass


def save_model(model, model_filepath):
    joblib.dump(model, 'filename.pkl') 
    
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format('disaster_response_pipeline_project\data\DisasterResponse.db'))
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