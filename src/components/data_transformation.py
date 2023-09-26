from src.constants import *
from src.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.Utils import save_object



class FeatureEngineering:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # The fit method is used to perform any necessary computations or training.
        return self

    def Cleaning_Dataset(self, df):
        try:
            """
            This method is responsible for cleaning data, like removing unwanted or correlated data.
            """
            excluded_columns = ['Customer Id', 'Artist Name', 'Width', 'Weight', 'Scheduled Date', 'Delivery Date', 'Customer Location']
            cleaned_df = df.drop(excluded_columns, axis=1)
            return cleaned_df
        except Exception as e:
            raise CustomException(e)

    def remove_outliers(self, df):
        try:
            numerical_cols = df.select_dtypes(include=np.number).columns
            for col in numerical_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                df[col] = np.where(
                    (df[col] >= lower_limit) & (df[col] <= upper_limit),
                    df[col],
                    np.nan
                )
            return df
        except Exception as e:
            raise CustomException(e)

    def transform(self, X):
        try:
            cleaned_df = self.Cleaning_Dataset(X)
            cleaned_df_no_outliers = self.remove_outliers(cleaned_df)
            return cleaned_df_no_outliers
        except Exception as e:
            raise CustomException(e)

    
@dataclass
class DataTransformationConfig():
  processor_object_path=TRANSFORMER_OBJECT_FILE
  feature_engineering_object_path=FEATURE_ENGINEERING_OBJECT_FILE
  feature_eng_train=FE_TRAIN_DATA_PATH
  feature_eng_test=FE_TEST_DATA_PATH
  transformed_train=TRANSFORMED_TRAIN_FILE
  transformed_test=TRANSFORMED_TEST_FILE

class DataTransformation:
  def __init__(self):
    self.data_transformation_config=DataTransformationConfig()

  def get_preprocessor_object(self):
    """
    this method only returns preprocessor object 
    """
    try:
      logging.info('crreating categorical pipeline')
      cat_pipe = Pipeline([
            ('imputation', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False))
        ])
      logging.info('crating numerical pipeline')
      num_pipe = Pipeline([
            ('imputation', SimpleImputer(strategy='mean')),
            ('scaling', StandardScaler())
        ])

      num_cols = ['Artist Reputation', 'Height', 'Price Of Sculpture', 'Base Shipping Price']
      cat_cols =['Material', 'International', 'Express Shipment',
       'Installation Included', 'Transport', 'Fragile', 'Customer Information',
       'Remote Location']
      logging.info('combining num_pipe line and cat_pipeline as preprocessor pipeline')
      preprocessor = ColumnTransformer([
            ('cat_pipe', cat_pipe, cat_cols),
            ('num_pipe', num_pipe, num_cols)
        ], remainder='passthrough')
      logging.info('crated preprocessor pipeline ')

      return preprocessor

    except Exception as e:
      raise CustomException(e, sys)
  def get_feature_engineering_object(self):
    """
    this method is used for creating an instance of FeatureEngineering class
    """
    try:    
        feature_engineering_object = Pipeline([
        ('fe', FeatureEngineering())
    ])
        return feature_engineering_object
    except Exception as e:
        raise CustomException(e, sys)

  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)
      logging.info('getting feature engineering object in initiate data transformsation')
      fe_obj=self.get_feature_engineering_object()
      os.makedirs(os.path.dirname(self.data_transformation_config.feature_engineering_object_path),exist_ok=True)
      save_object(
        self.data_transformation_config.feature_engineering_object_path,fe_obj
      )

      logging.info('applying feature engineering pipeline in initiate data transformation')

      train_df_fe=fe_obj.fit_transform(train_df)
      os.makedirs(os.path.dirname(self.data_transformation_config.feature_eng_train),exist_ok=True)
      train_df_fe.to_csv(self.data_transformation_config.feature_eng_train,index=False)
      logging.info('feature engineered train data stored in Artifact/Data Transformation/FE data')
      
      test_df_fe=fe_obj.transform(test_df)
      os.makedirs(os.path.dirname(self.data_transformation_config.feature_eng_test),exist_ok=True)
      test_df_fe.to_csv(self.data_transformation_config.feature_eng_test,index=False)
      logging.info('feature engineered test data stored in Artifact/Data Transformation/FE data')
      logging.info('applying feature engineering pipeline in initiate data transformation compleeted')
      preprocessor_obj=self.get_preprocessor_object()
      os.makedirs(os.path.dirname(self.data_transformation_config.processor_object_path),exist_ok=True)
      save_object(self.data_transformation_config.processor_object_path,preprocessor_obj)
      target_column='Cost'
      X_train = train_df_fe.drop(target_column, axis=1)
      X_train=preprocessor_obj.fit_transform(X_train)
      X_test = test_df_fe.drop(target_column, axis=1)  
      X_test=preprocessor_obj.transform(X_test)
      y_train = train_df_fe[target_column]
      y_test = test_df_fe[target_column]
    
   
      train_arry = np.concatenate((X_train, np.array(y_train).reshape(-1, 1)), axis=1)
      test_arry = np.concatenate((X_test, np.array(y_test).reshape(-1, 1)), axis=1)

      train_df=pd.DataFrame(train_arry)
      test_df=pd.DataFrame(test_arry)
      os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train),exist_ok=True)
      train_df.to_csv(self.data_transformation_config.transformed_train,index=False)
      logging.info('preproceesed train  data stored in Artifacts/DataTransformation/Transformed data/Transformed_train.csv')
      os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test),exist_ok=True)
      test_df.to_csv(self.data_transformation_config.transformed_test)
      logging.info('preproceesed test data stored in Artifacts/DataTransformation/Transformed data/Transformed_test.csv')



      return  X_train,X_test,y_train,y_test

    except Exception as e:
      raise CustomException(e, sys)
        
