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


FE_TRAIN_DATA_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,FE_DATA_FOLDER,FE_TRAIN)

                                            
FE_TEST_DATA_PATH=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,FE_DATA_FOLDER,FE_TEST)
                                            
TRANSFORMED_TRAIN_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,TRANSFORMED_DATA_FOLEDR,TRANSFORMED_TRAIN)

TRANSFORMED_TEST_FILE=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,DATA_TRANSFORMATION_FOLDER,TRANSFORMED_DATA_FOLEDR,TRANSFORMED_TEST)

#model training

MODEL_OBJECT=os.path.join(ROOT_DIR,ARTIFACT_FOLDER,MODEL_TRAINING,MODEL_OBJECT)
#**** app data****

#batch prediction 
BATCH_INPUT_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,BATCH_INPUT_FOLDER,CURRENTR_TIME_STAMP,INPUT_CSV)
BATCH_OUTPUT_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,BATCH_OUTPUT_FOLDER,CURRENTR_TIME_STAMP,OUTPUT_CSV)
BATCH_FE_DATA_FILE=os.path.join(ROOT_DIR,PREDICTION_FOLDER,BATCH_FE_FOLDER,CURRENTR_TIME_STAMP,BATCH_FE_DATA)
