from src.constants import *
from src.config.configuration import *
import os,sys
from src.logger import logging
from src.exception import CustomException
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path,obj):
    try:
       os.makedirs(os.path.dirname(file_path), exist_ok=True)
       with open(file_path, "wb") as file:
                pickle.dump(obj, file)

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(x_train, y_train, x_test, y_test, models,param ):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,cv=3,param_grid=para)
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            #model.fit(x_train, y_train)  # Train model

            y_train_pred = model.predict(x_train)

            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_model(file_path):
     
     try:
        with open(file_path,'rb') as f:
            object=pickle.load(f)
            return object
     except Exception as e:
        raise CustomException(e)