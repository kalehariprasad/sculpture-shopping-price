from src.constants import *
from src.config.configuration import *
import os,sys
from typing import Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.Utils import save_object


class FeatureEngineering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        try:
            logging.info('featutre engineering dropping columns outlier capping started')
            excluded_columns = ['Customer Id', 'Artist Name', 'Width', 'Weight', 'Customer Location', 'Scheduled Date', 'Delivery Date']
            cleaned_df = X.drop(excluded_columns, axis=1)

            numerical_cols = cleaned_df.select_dtypes(include=np.number).columns
            for col in numerical_cols:
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                cleaned_df[col] = np.where(
                    cleaned_df[col] > upper_limit,
                    upper_limit,
                    np.where(
                        cleaned_df[col] < lower_limit,
                        lower_limit,
                        cleaned_df[col]
                    )
                )
            logging.info('featutre engineering dropping columns outlier capping compleeted')
            return cleaned_df

        except Exception as e:
            raise CustomException(e, sys)


@dataclass
class DataTransformationConfig():
    processor_object_path = TRANSFORMER_OBJECT_FILE
    feature_engineering_object_path = FEATURE_ENGINEERING_OBJECT_FILE
    feature_eng_train = FE_TRAIN_DATA_PATH
    feature_eng_test = FE_TEST_DATA_PATH
    transformed_train = TRANSFORMED_TRAIN_FILE
    transformed_test = TRANSFORMED_TEST_FILE


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor(self):
        try:
          logging.info('categorical pipeline started')
          cat_pipe = Pipeline([
              ('imputation', SimpleImputer(strategy='most_frequent')),
              ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
          ])
          logging.info('numerical  pipeline complleted')
          num_pipe = Pipeline([
              ('imputation', SimpleImputer(strategy='mean')),
              ('scaling', StandardScaler())
          ])
          logging.info('numerical pipeline compleeted')

          num_cols = ['Artist Reputation', 'Height', 'Price Of Sculpture', 'Base Shipping Price']
          cat_cols = ['Material', 'International', 'Express Shipment', 'Installation Included', 'Transport', 'Fragile', 'Customer Information', 'Remote Location']

          preprocessor = ColumnTransformer([
              ('cat_pipe', cat_pipe, cat_cols),
              ('num_pipe', num_pipe, num_cols)
          ], remainder='passthrough')
          logging.info('combined both categorical and numerical pipelines as preprocessor ')

          return preprocessor
        except Exception as e:
          raise CustomException(e, sys)       

    def get_fe_obj(self):
        return FeatureEngineering()  # Changed this to call FeatureEngineering class

    def initiate_data_transformation(self, train_df_path, test_df_path):
        try:
          train_df=pd.read_csv(train_df_path)
          test_df=pd.read_csv(test_df_path)
          fe_obj = self.get_fe_obj()
          preprocessor_obj = self.get_preprocessor()
          logging.info('obtained both fe_obj and preprocessor obj')

          logging.info('applying feature engineering obj to both train data and test  data')
          train_df_fe = fe_obj.fit_transform(train_df)
          test_df_fe = fe_obj.transform(test_df)

          os.makedirs(os.path.dirname(self.data_transformation_config.feature_eng_train),exist_ok=True)
          train_df_fe.to_csv(self.data_transformation_config.feature_eng_train,index=False)
          os.makedirs(os.path.dirname(self.data_transformation_config.feature_eng_test),exist_ok=True)
          test_df_fe.to_csv(self.data_transformation_config.feature_eng_test,index=False)
          logging.info('stored both train and test data save after applying feature engineering object')

          logging.info('saving feature engineering object')
          os.makedirs(os.path.dirname(self.data_transformation_config.feature_engineering_object_path),exist_ok=True)
          save_object(
          self.data_transformation_config.feature_engineering_object_path,fe_obj
          )
          logging.info('feature enginnering object saved at Artifact/Data transformation /preprocessing')




          logging.info('applying feature processor obj to both train data and test  data')
          X_train = preprocessor_obj.fit_transform(train_df_fe)
          X_test = preprocessor_obj.transform(test_df_fe)
          
          logging.info('saving preprocessor obj')
          os.makedirs(os.path.dirname(self.data_transformation_config.processor_object_path),exist_ok=True)
          save_object(self.data_transformation_config.processor_object_path,preprocessor_obj)
          logging.info('preprocessor object saved at Artifact/Data transformation /preprocessing')
          logging.info('splitting target column for both train and test data as train_y ,test_y')
          target_column = 'Cost'
          y_train = train_df_fe[target_column]
          y_test = test_df_fe[target_column]

          logging.info('combining both train_x,test_x as train array and train_y ,test_y as test array')
          train_array = np.concatenate((X_train, np.array(y_train).reshape(-1, 1)), axis=1)
          test_array = np.concatenate((X_test, np.array(y_test).reshape(-1, 1)), axis=1)
          os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train),exist_ok=True)
          os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test),exist_ok=True)
          train_array_df=pd.DataFrame(train_array)
          test_array_df=pd.DataFrame(test_array)
          train_array_df.to_csv(self.data_transformation_config.transformed_train)
          test_array_df.to_csv(self.data_transformation_config.transformed_test)
          logging.info('stored both transformed train and test arrays  as Datafrmae at Artifact/Data transformation /transformed data ')

          return train_array, test_array
        except Exception as e:
          raise CustomException(e, sys)         

