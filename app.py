from src.Student_Performance_ML_Project.logger import logging
from src.Student_Performance_ML_Project.exception import CustomException
from src.Student_Performance_ML_Project.components.data_ingestion import DataIngestion
import sys




if __name__ == "__main__":
    logging.info("Execution Started")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initite_data_ingestion()
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)