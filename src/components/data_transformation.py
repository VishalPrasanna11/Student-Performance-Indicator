import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessed_data_path: str = os.path.join("artifacts", "proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = ['reading score', 'writing score']
            categorical_features = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
                ]
            num_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('std_scaler', StandardScaler())
                 ]
             )
            cat_pipeline = Pipeline(
            steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False)),  # Ensure that OneHotEncoder returns dense matrices
                ('scaler', StandardScaler(with_mean=False))  # Use with_mean=False for sparse matrices
            ]
            )  
            logging.info("Data Transformation process started")
            logging.info(f'Numeric pipeline: {num_pipeline}')
            logging.info(f'Categorical pipeline: {cat_pipeline}')
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', num_pipeline, numeric_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.error("Data Transformation process failed")
            raise CustomException(e, sys)

    def initiate_data_transformation(self,train_path, test_path):
        try:
           trained_df= pd.read_csv(train_path)
           test_df= pd.read_csv(test_path)  

           logging.info("Data Transformation process started")
           preprocessor = self.get_data_transformer_object()
           target_column = 'math score'
           numerical_columns = ['reading score', 'writing score']

           input_features_train_df =trained_df.drop(target_column, axis=1)
           target_features_train_df = trained_df[target_column]

           input_features_test_df = test_df.drop(target_column, axis=1)
           target_features_test_df = test_df[target_column]

           input_features_train_df = preprocessor.fit_transform(input_features_train_df)
           input_features_test_df = preprocessor.transform(input_features_test_df)

           logging.info("Data Transformation process completed successfully")
           
           train_arr = np.c_[
               input_features_train_df, np.array(target_features_train_df)
           ]
           test_arr = np.c_[
                input_features_test_df, np.array(target_features_test_df)
              ]
           logging.info("Preprocessed data saved successfully")

           save_object(
               
               file_path = self.transformation_config.preprocessed_data_path,
               obj = preprocessor

           )
           return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessed_data_path
           )
        except Exception as e:
            logging.error("Data Transformation process failed")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    train_data, test_data = obj.initiate_data_transformation()

    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
    
