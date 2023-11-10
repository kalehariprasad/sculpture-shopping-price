from src.constants import *
from src.config.configuration import *
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.Utils import save_object,model_evaluation
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


@dataclass
class Model_TrainerConfig:
    model_path=MODEL_OBJECT 


class Model_Trainer:
    def __init__(self):
        self.model_trainer_config=Model_TrainerConfig()
    def initiate_model_training(slef,train_arry,test_arry):
        try:
            X_train=train_arry[:,:-1]
            y_train=train_arry[:,-1]
            X_test=test_arry[:,:-1]
            y_test=test_arry[:,-1]
            models={
                'LinearRegression': LinearRegression(),
                'SVR' : SVR(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor()
            }
            param_grid = {
                "linear Regression": {},
                "SVR": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"]
                },
                "Random Forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Decision Tree": {
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "ExtraTreesRegressor": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10]
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5],
                    "learning_rate": [0.01, 0.1]
                }
            }
            best_model,best_model_score=model_evaluation(X_train=X_train, y_train=y_train,
                                                          X_test=X_test, y_test=y_test,
                                                           models=models, param_grids=param_grid)
            
            logging.info(f'best model is :{best_model} with score of :{best_model_score}')
            save_object(file_path=self.model_trainer_config.model_path,obj=best_model)
            logging.info(f'modeel save to artifact/model training /model')
            print(f'best model is :{best_model} with score of :{best_model_score}')
            return best_model,best_model_score

            
        except Exception as e:
            raise CustomException(e)
