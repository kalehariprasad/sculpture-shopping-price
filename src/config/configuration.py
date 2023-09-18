import os,sys
from src.constants import *

ROOT_DIR=ROOT_DIR
#data injection configuration
API_FETHED_DATA=os.path.join(ROOT_DIR,DATA_DIR_FOLDER)
CURRENT_DATA_PATH=os.path.join(ROOT_DIR,CURRENT_DATA_FOLDER,CURRENT_DATA_FILE)
RAW_TRAIN_DATASET_PATH=os.path.join(ROOT_DIR,DATA_DIR_FOLDER,DATA_DIR_TRAIN)
RAW_TEST_DATASET_PATH=os.path.join(ROOT_DIR,DATA_DIR_FOLDER,DATA_DIR_TEST)
RAW_SS_DATASET_PATH=os.path.join(ROOT_DIR,DATA_DIR_FOLDER,DATA_DIR_SS)



RAW_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_INJECTION_FOLDER,
                           DATA_INJECTION_RAW_DATA_FOLDER,DATA_INJECTION_INJECTED_DATA_FILE)

TRAIN_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_INJECTION_FOLDER,
                            DATA_INJECTION_INJECTED_DATA_FOLDER,INJECTED_TRAIN_DATA)

TEST_FILE_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_INJECTION_FOLDER,
                            DATA_INJECTION_INJECTED_DATA_FOLDER,INJECTED_TEST_DATA)

#data transformation configuration

TRANSFORMER_OBJECT_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,PRE_PROCESSING_FOLDER,
                                        PRE_PROCESSING_OBJECT)

FEATURE_ENGINEERING_OBJECT_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,
                                            PRE_PROCESSING_FOLDER,FEATURE_ENGINEERING_OBJECT)
                                            
TRANSFORMED_TRAIN_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,
                                    TRANSFORMED_TRAIN)

TRANSFORMED_TEST_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,
                                    TRANSFORMED_TEST)