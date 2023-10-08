import os,sys
from datetime import datetime

def get_currentr_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENTR_TIME_STAMP=get_currentr_time_stamp()

ROOT_DIR=os.getcwd()
CURRENT_DATA_FOLDER='data'
CURRENT_DATA_FILE='train.csv'
DATA_DIR_TRAIN='Raw train.csv'
DATA_DIR_TEST='Raw test.csv'
DATA_DIR_SS='saple submission.csv'
DATA_DIR_FOLDER='API throgh fetched data'
ARTIFACT_FOLDER='Artifact'
DATA_INJECTION_FOLDER='Data injection'
DATA_INJECTION_RAW_DATA_FOLDER='Raw data'
DATA_INJECTION_INJECTED_DATA_FOLDER='Injected train & test data'
DATA_INJECTION_INJECTED_DATA_FILE='injected raw file.csv'
INJECTED_TRAIN_DATA='train.csv'
INJECTED_TEST_DATA='test.csv'
## data transformation variables

DATA_TRANSFORMATION_FOLDER='Data trransformation'
PRE_PROCESSING_FOLDER='Preprocessing'
PRE_PROCESSING_OBJECT='preprocessor.pkl'
FEATURE_ENGINEERING_OBJECT='feature_enginrrting.pkl'
FE_DATA_FOLDER='Feature engineered data'
FE_TRAIN='Fe_train.csv'
FE_TEST='Fe_test.csv'
TRANSFORMED_DATA_FOLEDR='Transformed data'
TRANSFORMED_TRAIN='Transformed_train.csv'
TRANSFORMED_TEST='Transformed_test.csv'

#model training
MODEL_TRAINING="Model training"
MODEL_OBJECT='model.pkl'
