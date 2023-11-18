from src.constants import *
from src.config.configuration import *
import os,sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
import pickle
from src.Utils import load_model
from sklearn.pipeline import Pipeline
from typing import Any

class BatchPredictionConfig:
    batch_input_file=BATCH_INPUT_FILE
    batch_fe_file=BATCH_FE_DATA_FILE
    batch_output=BATCH_OUTPUT_FILE
    fe_file_path=FEATURE_ENGINEERING_OBJECT_FILE
    transformer_file_path=TRANSFORMER_OBJECT_FILE
    model_file_path=MODEL_OBJECT
    train_file=TRAIN_FILE_PATH
class BatchPrediction:
    def __init__(self):
        self.batchpredictionconfig = BatchPredictionConfig()

    def main(self,input_df):
    
        try:
            with open(input_path, 'rb') as f:
                input_df = pd.read_csv(f)  
                print("Loaded input data:", input_df.head())  

            with open(self.batchpredictionconfig.fe_file_path,'rb') as f:
                feature_pipeline=pickle.load(f)

            with open(self.batchpredictionconfig.transformer_file_path,'rb')as f:
                transformer=pickle.load(f)

            model=load_model(self.batchpredictionconfig.model_file_path)
            
      
      
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_input_file),exist_ok=True)
            input_df.to_csv(self.batchpredictionconfig.batch_input_file)
   

            fe_df=feature_pipeline.transform(input_df)
            print(fe_df.columns)
     
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_fe_file),exist_ok=True)
            fe_df.to_csv(self.batchpredictionconfig.batch_fe_file)

            
            transformed_df=transformer.transform(fe_df)
            

            prediction=model.predict (transformed_df)
            fe_df['prediction']=prediction
            output_df=fe_df[['Artist Reputation', 'Height', 'Material', 'Price Of Sculpture',
       'Base Shipping Price', 'International', 'Express Shipment',
       'Installation Included', 'Transport', 'Fragile', 'Customer Information',
       'Remote Location', 'prediction']]
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_output),exist_ok=True)
            fe_df.to_csv(self.batchpredictionconfig.batch_output)
            
            return output_df
    
        except Exception as e:
            raise CustomException(e,sys)

    
input_path=TEST_FILE_PATH
if __name__ == "__main__":
    obj = BatchPrediction()
    prediction=obj.main(input_path)
    print('Predicted df:', prediction)
