import os 
import sys 
import pandas as pd 
import numpy as np 

from sklearn.metrics import r2_score

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

def evaluate_model(X_train, y_train, X_test, y_test, models):
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
        model_report = {}
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)  
            y_pred = model.predict(X_test)  
            score = r2_score(y_test, y_pred)  
            print(f"Model {model_name} : {score*100}")
            model_report[model_name] = score
        
        return model_report
    except Exception as e:
        raise CustomException(e,sys)
    