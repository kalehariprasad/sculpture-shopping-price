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
        raise CustomException(e)
    

def model_evaluation(X_train, y_train, X_test, y_test, models, param_grids):
    report = {}

    for model_name, model in models.items():
        param_grid = param_grids.get(model_name, {})  
        grid_search = GridSearchCV(model, param_grid, scoring='r2', cv=5)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_  
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)


        report[model_name] = test_model_score
        best_model_name = max(report, key=report.get)
        best_model = models[best_model_name]
        best_model_score = report[best_model_name]
        

        return best_model,best_model_score
