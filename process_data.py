import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on= 'id')
    return df


def clean_data(df):
    df = df.drop_duplicates()
    categories = df['categories'].str.split(';|-')
    cols=df.categories[:1]
    cols=cols.str.replace('\d',"")
    cols=cols.str.replace('-',"")
    cols=cols.str.split(";")
    categories = df.categories.str.split(pat=";",expand=True)
    categories.columns=cols.tolist()[0]
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)
    
        categories[column] = pd.DataFrame(categories[column].astype(int))
    df.drop(['categories'],axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)
    df['related'].replace(2,1,inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('df1', engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df1, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
