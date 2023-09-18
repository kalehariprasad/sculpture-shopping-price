from src.constants import *
from src.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException


class Distribution_Outliers:
  def __init__(self):
    pass

  def Distribution(self,df):  
    """
    this method is responsible for checking  unique values in  categorical variable and 
    distribution for Each numerical variable 
    """
    try:
      logging.info('feature engineering has stared')

      for col in df.columns:
          if df[col].dtype == 'object':

              print(f"Value counts for column '{col}':")
              print(df[col].value_counts())
              print("\n")
          else:

              plt.figure(figsize=(8, 6))
              sns.histplot(x=col, bins=20,data=df, kde=True)
              plt.title(f"Distribution of '{col}'")
              plt.xlabel(col)
              plt.ylabel("Frequency")
              plt.show()
  
    
    except Exception as e:
          raise CustomException(e, sys)

  def detect_and_visualize_outliers(self,df): 
      try:
        """
        this method is responsible for detecting and visualising outliers in numerical columns
        """
        logging.info("detecting and visualing outliers")
        for col in df.columns:
          if  df[col].dtype != 'object' and df[col].dtype != 'datetime64[ns]':
              plt.figure(figsize=(8, 6))

              sns.boxplot(x=df[col])

              plt.title(f"Outliers in '{col}'")
              plt.xlabel(col)
              plt.show()
    
        
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