import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """Loads messages and categories data from provided paths, 
    using pandas read_csv methon, and performs inner merge on "id" key.
    """
    
    # loading messages dataset 
    messages = pd.read_csv(messages_filepath)

    # loading categories dataset 
    categories = pd.read_csv(categories_filepath)

    # inner merging two datasets using "id" column as a key
    df = messages.merge(categories, how='inner', on=['id'])
    
    return df


def clean_data(df):
    """Performs cleaning of provided dataframe by extracting categories 
    into separate columns and dropping duplicates.

    Args:
    df: pandas.DataFrame. Dataframe with messages and categories.

    Returns:
    df: pandas.DataFrame. Cleaned dataframe with categories splitted into separate columns.
    """

    # expanding categories into separate clumns using split method
    categories = df['categories'].str.split(pat=";",expand=True)

    # selecting the first row of the categories dataframe for categories names extraction
    row = categories.iloc[0]

    # assigning new column names for categories by applying lambda function
    category_colnames = row.apply(lambda x: x.split("-")[0]).tolist()
    categories.columns = category_colnames

    for column in categories:
        # setting each value to be the last character of the string 
        categories[column] = categories[column].astype(str).apply(lambda x: x.split("-")[1])
    
        # converting column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # dropping old categories columns and concatenating with new categories dataframe
    df.drop(["categories"], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    # dropping duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):    
    """Saves clean dataframe into an sqlite database."""

    # creating sqllite engine
    engine = create_engine('sqlite:///'+database_filename)
    
    # saving dataframe into sqlite database 
    df.to_sql('MessagesCategorized', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
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