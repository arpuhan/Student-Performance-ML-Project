import sys
from dataclasses import dataclass

import numpy as np
import pandas as Pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.Student_Performance_ML_Project.exception import CustomException
from src.Student_Performance_ML_Project.logger import logging
import os

#path saving for pkl file
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transform_object(self):
        '''
        This function is responsible for data trasformation
        '''
        try:
            numerical_columns=["writing score","reading score"]
            categorical_columns=[
                "gender",
                "race/ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(steps=[
                '''
                this helps to handle missing values if new value is added to the data set'''
                ("imputer",SimpleImputer(strategy='median')),
                ("scalar",StandardScaler)
            ])
            cat_pipeline=Pipeline(steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder)
                ("scalar",StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            preprocessing_obj=self.get_data_transform_object()

            target_column_name="math score"
            numerical_columns=["writing score","reading score"]

            # divide the train dataset to independent and dependent feature
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            taget_feature_train_df=train_df[target_column_name]

            # divide the test dataset to independent and dependent feature
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            taget_feature_test_df=test_df[target_column_name]

            logging.info("Reading the train and test file")
        except Exception as e:
            raise CustomException(e,sys)