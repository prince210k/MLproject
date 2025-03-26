import os 
import sys 
import pandas as pd 
import numpy as np 

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
import dill 

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)  

def evaluate_model(X_train, y_train, X_test, y_test, models,param):
    try:
        """
        Trains and evaluates multiple regression models.
        
        Parameters:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training target values
        X_test (numpy.ndarray): Testing features
        y_test (numpy.ndarray): Testing target values
        models (dict): Dictionary of models to train and evaluate
        
        Returns:
        dict: A dictionary containing model names and their corresponding R^2 scores on the test set
        """
        model_scores = {}
    
        for model_name, model in models.items():
            if not param.get(model_name):
                model.fit(X_train, y_train)
            else:
                gs = GridSearchCV(
                    model, 
                    param[model_name], 
                    cv=3, 
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                model = gs.best_estimator_  

            y_pred = model.predict(X_test)
            print(f"Model {model_name} : {r2_score(y_test, y_pred)*100}")
            model_scores[model_name] = r2_score(y_test, y_pred)
        
            return model_scores

    except Exception as e:
        raise CustomException(e,sys)
    
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except:
        raise CustomException(e,sys)