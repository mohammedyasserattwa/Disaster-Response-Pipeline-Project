import pandas as pd
from sqlalchemy import create_engine

def load_data(database_filepath):
    """
    Loads data from SQLite database.

    Parameters:
    database_filepath: Filepath to the database

    Returns:
    X: Features
    Y: Target
    """
    # Load data from the database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_messages", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X, Y
