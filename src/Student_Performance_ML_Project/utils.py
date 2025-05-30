import os
import sys
from src.Student_Performance_ML_Project.exception import CustomException
from src.Student_Performance_ML_Project.logger import logging
import pandas as pd
from dotenv import load_dotenv
import pymysql

load_dotenv()

host =os.getenv('host')
user =os.getenv('user')
password =os.getenv('password')
db =os.getenv('db')

def read_sql_data():
    logging.info("Reading SQL Database Started")
    try:
        mydb = pymysql.connect(
            host=host,
            user =user,
            password=password,
            db =db
        )
        logging.info("Connection Established",mydb)

        df= pd.read_sql_query('Select * from students_performance',mydb)
        print(df.head())

        return df
    
    except  Exception as ex:
        raise CustomException(ex,sys)