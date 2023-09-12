import pandas as pd

class DataLoader:
    def load_data(self, messages_filepath, categories_filepath):
        """
        Loads and merges datasets from two filepaths.

        Parameters:
        messages_filepath (str): path to messages CSV dataset
        categories_filepath (str): path to categories CSV dataset

        Returns:
        df (DataFrame): merged dataset
        """
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath)
        df = messages.merge(categories, how='outer', on=['id'])
        return df
    
    def clean_data(self, df):
        """
        Cleans the DataFrame.

        Parameters:
        df (DataFrame): merged dataset

        Returns:
        df (DataFrame): cleaned dataset
        """
        categories = df['categories'].str.split(';', expand=True)
        row = categories.head(1)
        category_colnames = row.applymap(lambda x: x[:-2]).iloc[0, :]
        categories.columns = category_colnames

        for column in categories:
            categories[column] = categories[column].astype(str).str[-1]
            categories[column] = categories[column].astype(int)

        categories['related'] = categories['related'].replace(to_replace=2, value=1)

        df.drop(columns=['categories'], axis=1, inplace=True)
        df = pd.concat([df, categories], axis=1)
        df.drop_duplicates(inplace=True)

        return df
    
    def save_data(self, df, database_filename):
        """
        Saves the DataFrame to a SQLite database.

        Parameters:
        df (DataFrame): cleaned dataset
        database_filename (str): path to SQLite database
        """
        from sqlalchemy import create_engine
        engine = create_engine('sqlite:///{}'.format(database_filename))
        df.to_sql('disaster_messages', engine, index=False, if_exists='replace')