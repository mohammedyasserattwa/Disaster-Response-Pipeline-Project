import sys
import logging
import coloredlogs
from data_loader import load_data
from model_builder import build_model, evaluate_model, save_model
from sklearn.model_selection import train_test_split

# Configure logging
coloredlogs.install(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    level="INFO",
    logger=logging.getLogger(),
)

def main():
    """Builds the model, trains the model, evaluates the model, and saves the model."""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        # Log start of the process
        logging.info("Starting the process...")
        
        try:
            logging.info('Loading data...\n    DATABASE: %s', database_filepath)
            X, Y = load_data(database_filepath)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

            logging.info('Building model...')
            model = build_model()

            logging.info('Training model...')
            model.fit(X_train, Y_train)

            logging.info('Evaluating model...')
            evaluate_model(model, X_test, Y_test)

            logging.info('Saving model...\n    MODEL: %s', model_filepath)
            save_model(model, model_filepath)

            logging.info('Trained model saved!')
        
        except Exception as e:
            # Log any exceptions
            logging.error("An error occurred: %s", str(e))
        
        # Log completion of the process
        logging.info("Process completed.")
        
    else:
        logging.error('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'main.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
