# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def main():

    # load messages dataset
    messages = pd.read_csv(sys.argv[1])

    # load categories dataset
    categories = pd.read_csv(sys.argv[2])

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda item: item[:-2])

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df.drop(columns=['categories'], inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    engine = create_engine('sqlite:///' + sys.argv[3])
    df.to_sql('InsertTableName', engine, index=False)
   
if __name__ == "__main__":
    main()