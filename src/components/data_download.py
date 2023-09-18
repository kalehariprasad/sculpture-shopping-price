import os
import sys
import opendatasets as od
from src.constants import *
from src.config.configuration import *
import pandas as pd
import numpy as np
import shutil
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

class DataDownloadConfig:
    raw_train_file_path: str = RAW_TRAIN_DATASET_PATH
    raw_test_file_path: str = RAW_TEST_DATASET_PATH
    sample_submission: str = RAW_SS_DATASET_PATH
    raw_file_path: str = RAW_FILE_PATH
    train_file_path: str = TRAIN_FILE_PATH
    test_file_path: str = TEST_FILE_PATH
    api_fetcehd_folder: str = API_FETHED_DATA
    api_username = "kalehariprasad"
    api_key = "9a694c8c79d80c1dff52f1e065905992"

class DataDownload:
    def __init__(self):
        self.data_download_config = DataDownloadConfig()

    def data_download(self, dataset_name):
        try:
            logging.info('Started downloading dataset')

            # Specify the directory where you want to download the dataset
            download_dir = self.data_download_config.api_fetcehd_folder

            # Use opendatasets to download the dataset
            od.download(dataset_name, path=download_dir, unzip=True)

            logging.info('Created directories for each file')
            os.makedirs(self.data_download_config.raw_train_file_path, exist_ok=True)
            os.makedirs(self.data_download_config.raw_test_file_path, exist_ok=True)
            os.makedirs(self.data_download_config.sample_submission, exist_ok=True)

            # Move the downloaded files to their respective directories
            shutil.move(
                f"{self.data_injection_config.raw_file_path}/train.csv",
                self.data_injection_config.raw_train_file_path
            )
            logging.info('raw train file downloaded')
            shutil.move(
                f"{self.data_injection_config.raw_file_path}/train.csv",
                self.data_injection_config.raw_test_file_path
            )
            logging.info('Raw test file downloaded')
            shutil.move(
                f"{self.data_injection_config.raw_file_path}/sample_submission.csv",
                self.data_injection_config.sample_submission
            )
            logging.info('sample submission downloadedd')
           
            return (
                self.data_download_config.raw_train_file_path,
                self.data_download_config.raw_test_file_path,
                self.data_download_config.sample_submission
            )

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataDownload()
    dataset_name = "https://www.kaggle.com/code/klmsathishkumar/shipping-cost-prediction/input"  
    raw_train_path, raw_test_path, sample_submission_path = obj.data_download(dataset_name)
