from src.constants import *
from src.config.configuration import *
import os,sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_injection import DataInjection
from src.components.data_transformation import DataTransformation
from src.components.model_training import Model_Trainer

class Train:
    def __init__(self):
        self.c=0
        print(f"**************{self.c}******************")

    def main(self):
        obj = DataInjection()
        train_path, test_path=obj.Initiate_data_injection() 
        transform_object=DataTransformation()
        train_arry,test_arry= transform_object.initiate_data_transformation(train_path, test_path)
        model_train=Model_Trainer()
        model_train.initiate_model_training(train_arry,test_arry)