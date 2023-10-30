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
class BatchPredictionConfig:
    batch_input_file=BATCH_INPUT_FILE
    batch_fe_file=BATCH_FE_DATA_FILE
    batch_output=BATCH_OUTPUT_FILE
    fe_file_path=FEATURE_ENGINEERING_OBJECT_FILE
    transformer_file_path=TRANSFORMER_OBJECT_FILE
    model_file_path=MODEL_OBJECT
    train_file=TRAIN_FILE_PATH
class BatchPrediction:
    def __init__(self,):
        
        self.batchpredictionconfig=BatchPredictionConfig()

    def main(self,input_file_path):
        self.input_file_path=input_file_path
        try:


            with open(self.batchpredictionconfig.fe_file_path,'rb') as f:
                feature_pipeline=pickle.load(f)

            with open(self.batchpredictionconfig.transformer_file_path,'rb')as f:
                transformer=pickle.load(f)

            model=load_model(self.batchpredictionconfig.model_file_path)
            print(f"Loaded model: {model}")
            feature_engineering_pipeline=Pipeline([
                ('feature engineering pipeline',feature_pipeline)
            ])
            df=pd.read_csv(self.input_file_path)
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_input_file),exist_ok=True)
            df.to_csv(self.batchpredictionconfig.batch_input_file)
            train_df=pd.read_csv(self.batchpredictionconfig.train_file)
         
            feature_engineering_pipeline.fit_transform(train_df)

            fe_df=feature_engineering_pipeline.transform(df)
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_fe_file),exist_ok=True)
            fe_df.to_csv(self.batchpredictionconfig.batch_fe_file)
            train_x = train_df.drop('Cost', axis=1)
            train_x_fe=feature_engineering_pipeline.transform(train_x)
            train_y = train_df['Cost']
            transformer.fit(train_x_fe)
            train_x_transformed=transformer.transform(train_x_fe)
               
            model.fit(train_x_transformed, train_y)

          
            fe_df.drop('Cost',axis=1,inplace=True)
            
            transformed_df=transformer.transform(fe_df)

            prediction=model.predict (transformed_df)
            fe_df['prediction']=prediction
            os.makedirs(os.path.dirname(self.batchpredictionconfig.batch_output),exist_ok=True)
            fe_df.to_csv(self.batchpredictionconfig.batch_output)



            return prediction
    
        except Exception as e:
            raise CustomException(e, sys)

    
input_path=TEST_FILE_PATH
if __name__ == "__main__":
    obj = BatchPrediction()
    prediction=obj.main(input_path)
    print('Predicted values:', prediction)
