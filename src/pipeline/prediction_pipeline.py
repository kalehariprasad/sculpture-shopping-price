from typing import Any
from src.constants import *
from src.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.components.data_injection import DataInjection
from src.components.data_transformation import DataTransformation
from src.components.model_training import Model_Trainer
from src.Utils import load_model
from dataclasses import dataclass

@dataclass
class predictpipelineConfig:
     
     transformer_file_path=TRANSFORMER_OBJECT_FILE
     model_file_path=MODEL_OBJECT
     train_file=FE_TRAIN_DATA_PATH
class PredictionPipeline:
     
     def __init__(self):
          self.predictionconfig=predictpipelineConfig()

     def prrdict(self,features):
          try:
            train_file=pd.read_csv(self.predictionconfig.train_file)
            preprocessor=load_model(self.predictionconfig.transformer_file_path)
            Model=load_model(self.predictionconfig.model_file_path)
            train_x = train_file.drop('Cost', axis=1)
            train_y = train_file['Cost']
            train_x_processed=preprocessor.fit_transform(train_x)
            Model.fit(train_x_processed,train_y)
            scaled_data=preprocessor.transform(features)
            prediction=Model.predict(scaled_data)

            return prediction
          except Exception as e:
            raise CustomException(e, sys)   

class CustomData:
    def __init__(self, Artist_Reputation:int, Height:float, Material:str,
                 Price_Of_Sculpture:int, Base_Shipping_Price:float,
                 International:str, Express_Shipment:str,
                 Installation_Included:str, Transport:str,
                 Fragile:str, Customer_Information:str,
                 Remote_Location:str):
        # Replace spaces and special characters with underscores
        self.Artist_Reputation = Artist_Reputation
        self.Height = Height
        self.Material = Material
        self.Price_Of_Sculpture = Price_Of_Sculpture
        self.Base_Shipping_Price = Base_Shipping_Price
        self.International = International
        self.Express_Shipment = Express_Shipment
        self.Installation_Included = Installation_Included
        self.Transport = Transport
        self.Fragile = Fragile
        self.Customer_Information = Customer_Information
        self.Remote_Location = Remote_Location
    def get_dataframe(self):
        try:
                 
            data_dict = {
                'Artist Reputation': [self.Artist_Reputation],
                'Height': [self.Height],
                'Material': [self.Material],
                'Price Of Sculpture': [self.Price_Of_Sculpture],
                'Base Shipping Price': [self.Base_Shipping_Price],
                'International': [self.International],
                'Express Shipment': [self.Express_Shipment],
                'Installation_Included': [self.Installation_Included],
                'Transport': [self.Transport],
                'Fragile': [self.Fragile],
                'Customer Information': [self.Customer_Information],
                'Remote Location': [self.Remote_Location]
            }
         
        
            df=pd.DataFrame(data_dict)
            column_mapping = {
            'Artist Reputation': 'Artist Reputation',
            'Height': 'Height',
            'Material': 'Material',
            'Price Of Sculpture': 'Price Of Sculpture',
            'Base Shipping Price': 'Base Shipping Price',
            'International': 'International',
            'Express Shipment': 'Express Shipment',
            'Installation_Included': 'Installation Included',
            'Transport': 'Transport',
            'Fragile': 'Fragile',
            'Customer Information': 'Customer Information',
            'Remote Location': 'Remote Location'
        }

            print("Keys in df:", df.columns)
            print("Keys in column_mapping:", column_mapping.keys())
            mapped_data = {column_mapping[streamlit_var]: df[streamlit_var] for streamlit_var in df}

            return mapped_data
            
        except Exception as e:
         raise CustomException(e, sys)           
    