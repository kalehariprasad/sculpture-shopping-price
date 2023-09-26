from src.constants import *
from src.config.configuration import *
import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle

def save_object(file_path,obj):
    try:
       os.makedirs(os.path.dirname(file_path), exist_ok=True)
       with open(file_path, "wb") as file:
                pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e)