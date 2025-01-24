import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        logging.info(f"Directory created or exists: {dir_path}")

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
         
        logging.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logging.error(f"Error in saving object: {e}")
        raise CustomException(e, sys)
    

    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
        

def evaluate_models(X_train,y_train,X_test,y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train) #model training with parameters

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            model_training_score = r2_score(y_train,y_train_pred)
            model_test_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = model_test_score

        return report
    
    except Exception as e:
        raise CustomException(e, sys)



