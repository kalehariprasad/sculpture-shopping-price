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





  

class DataTransformation:


    def makedir(self):
       FE_TRAIN_DATA_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,FE_DATA_FOLDER,FE_TRAIN)
       os.makedirs(os.path.dirname(FE_TRAIN_DATA_PATH),exist_ok=True)
       df=pd.read_csv(CURRENT_DATA_PATH)
       df.to_csv(FE_TRAIN_DATA_PATH)
       FE_TEST_DATA_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,FE_DATA_FOLDER,FE_TEST)
       os.makedirs(os.path.dirname(FE_TEST_DATA_PATH),exist_ok=True)
       df2=pd.read_csv(CURRENT_DATA_PATH)
       df2.to_csv(FE_TEST_DATA_PATH)


if __name__ == "__main__":
    data_transformation = DataTransformation()
    data_transformation.makedir()
