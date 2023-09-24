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


class FeatureEngineering:
  def __init__(self):
    pass
  def Cleaning_Dataset(self, df):
    try:

      """
      this method is resposible for cleanin data/ like removing unwanted or correlated data
      droping correlated data as of jupyter notebook  
      """
      cleaned_df = df.drop(['Customer Id', 'Artist Name', 'Width', 'Weight', 'Customer Location', 'Scheduled Date', 'Delivery Date'], axis=1)
        
      return cleaned_df

    except Exception as e:
      raise CustomException(e, sys)
      
  def outliers_removal(self,df): 
    try:
      """
      this method is responsible for capping outliers like <lower limit values with lower limit value
      and >upper_limit value with upper_limit value
      """
      logging.info('outliers capping has started')
      if isinstance(df, pd.DataFrame):

        for col in df.columns:

          if df[col].dtype != 'object' and df[col].dtype != 'datetime64[ns]':
              percentile25 = df[col].quantile(0.25)
              percentile75 = df[col].quantile(0.75)
              iqr = percentile75 - percentile25
              upper_limit = percentile75 + 1.5 * iqr
              lower_limit = percentile25 - 1.5 * iqr

              df[col] = np.where(
                  df[col] > upper_limit,
                  upper_limit,
                  np.where(
                      df[col] < lower_limit,
                      lower_limit,
                      df[col]
                  )
            )
        logging.info('outliers capping compleeted')
        return df
        
    except Exception as e:
      raise CustomException(e, sys)

  @staticmethod
  def fe_pipline():

    """
    This method is only responsible for creating a feature engineering pipeline.
    """
    try:
        df = pd.read_csv(CURRENT_DATA_PATH)
        fe_instance = FeatureEngineering()
        cleaned_df = fe_instance.Cleaning_Dataset(df=df)
        outlier = fe_instance.outliers_removal(df=cleaned_df)
        cleaned_df_pipeline = Pipeline([
          ('clean_df', cleaned_df)
        ])

        outlier_pipeline = Pipeline([
            ('outlier removel', outlier)
        ])

        feature_engineering_object = ColumnTransformer([
            ('clean_df', cleaned_df_pipeline, cleaned_df.columns), 
            ('outlier removel', outlier_pipeline, outlier.columns)  
        ], remainder='passthrough')
        
        return feature_engineering_object
    except Exception as e:
        raise CustomException(e, sys)


    except Exception as e:
      raise CustomException(e, sys)
    
@dataclass
class DataTransformationConfig():
  processor_object_path=TRANSFORMER_OBJECT_FILE
  feature_engineering_object_path=FEATURE_ENGINEERING_OBJECT_FILE
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
            ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first', sparse=False))
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
        
        fe_instance = FeatureEngineering()
        feature_engineering_object = fe_instance.fe_pipline()
        return feature_engineering_object
    except Exception as e:
        raise CustomException(e, sys)

  def initiate_data_transformation(self,train_path,test_path):
    try:
      train_df=pd.read_csv(train_path)
      test_df=pd.read_csv(test_path)
      logging.info(f'train_df 5 rows are :{train_df.head()}')
      logging.info(f'test_df 5 rows are :{test_df.head()}')
      logging.info('getting feature engineering object in initiate data transformsation')
      fe_obj=self.get_feature_engineering_object()
      logging.info('applying feature engineering pipeline in initiate data transformation')
      
      train_df_fe=fe_obj.fit_transform(train_df)
      test_df_fe=fe_obj.transform(test_df)
      logging.info('applying feature engineering pipeline in initiate data transformation compleeted')

      preprocessor_obj=self.get_preprocessor_object()

      target_column='Cost'
      X_train = train_df_fe.drop(target_column, axis=1)
      logging.info(f"Shape of train_df_fe: {train_df_fe.shape}")
      logging.info(f"Shape of X_train: {X_train.shape}")
      X_test = test_df_fe.drop(target_column, axis=1)  
      logging.info(f"Shape of test_df_fe: {test_df_fe.shape}")
      logging.info(f"Shape of X_test: {X_test.shape}")
      y_train = train_df_fe[target_column]
      y_test = test_df_fe[target_column]
      
      return  X_train,X_test,y_train,y_test

    except Exception as e:
      raise CustomException(e, sys)
        
