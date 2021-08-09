import sys

from sqlalchemy import create_engine
import pandas as pd


def load_data(messages_filepath: str, categories_filepath: str):
    """
    Loads and merges two csv files

    :param messages_filepath: path to the messages .csv file
    :param categories_filepath: path to the categories .csv file

    :return: a pandas DataFrame with merged columns from messages
            and categories  tables
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')

    return df


def clean_data(df: pd.DataFrame):
    """
    Takes DataFrame and splits category column into
    36 dummy variable columns

    :param df: DataFrame to be cleaned

    :return: new df of shape (26386, 40) with 36 new columns
    """
    # split category column into 36 features
    categories = df.categories.str.split(';', expand=True)

    # rename columns
    category_col_names = categories.loc[0].str.split('-').str[0]
    categories.columns = category_col_names

    # update values in target columns to integers 0-1
    for column in category_col_names:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # concat original df with categories df
    df_new = pd.concat([df, categories], axis=1)
    df_new.drop('categories', axis=1, inplace=True)

    # drop duplicates
    df_new = df_new.drop_duplicates()

    df_new['no_category'] = 0
    df_new.loc[df_new.iloc[:, 4:].sum(axis=1) == 0, 'no_category'] = 1

    # drop child_alone column as it has no values.
    df_new.drop(columns='child_alone', inplace=True)

    # unclear what 2 represents in this case.
    # drop 188 rows
    df_new = df_new[df_new.related != 2]

    return df_new


def save_data(df: pd.DataFrame, database_filename: str):
    """
    Loads the data into sql database

    :param df: DataFrame to be loaded
    :param database_filename: name of the database file to load the df in

    :return: None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


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
        print("Please provide the filepaths of the messages and categories \
              datasets as the first and second argument respectively, as \
              well as the filepath of the database to save the cleaned data \
              to as the third argument. \n\nExample: python process_data.py \
              disaster_messages.csv disaster_categories.csv \
              DisasterResponse.db")


if __name__ == '__main__':
    main()
