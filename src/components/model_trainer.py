import os 
import sys 

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR 
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trainerd_model_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and test data")
            
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]
            
            models = {
                "LinearRegression" : LinearRegression(),
                "SVR" : SVR(),
                "K-neighbors" : KNeighborsRegressor(),
                "DecisionTree" : DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "CatBoost" : CatBoostRegressor(verbose=False),
                "AdaBoost" : AdaBoostRegressor(),
                "GradientBoost" : GradientBoostingRegressor(),
                "XgBoost" : XGBRegressor()
            }
            param_grids = {
            # "SVR": {
            #     "kernel": ["linear", "rbf"],
            #     "C": [0.1, 1, 10],
            #     "epsilon": [0.01, 0.1]
            # },
            # "K-neighbors": {
            #     "n_neighbors": [3, 5, 7],
            #     "weights": ["uniform", "distance"],
            #     "p": [1, 2]
            # },
            # "DecisionTree": {
            #     "max_depth": [None, 5, 10],
            #     "min_samples_split": [2, 5],
            #     "min_samples_leaf": [1, 2]
            # },
            # "RandomForest": {
            #     "n_estimators": [50, 100],
            #     "max_depth": [None, 5, 10],
            #     "min_samples_split": [2, 5]
            # },
            # "CatBoost": {
            #     "iterations": [100, 200],
            #     "depth": [4, 6],
            #     "learning_rate": [0.01, 0.1]
            # },
            # "AdaBoost": {
            #     "n_estimators": [50, 100],
            #     "learning_rate": [0.01, 0.1, 1.0]
            # },
            # "GradientBoost": {
            # "n_estimators": [50, 100, 200,500],       # Number of boosting stages
            # "learning_rate": [ 0.1,0.5],   # Shrinkage (lower = more robust)
            # "max_depth": [2,3, 4, 5, 6],             # Max tree depth
            # "min_samples_split": [2, 5, 10],       # Min samples to split a node
            # "loss": ["squared_error", "absolute_error"],  # Loss function
            # "criterion": ["friedman_mse", "squared_error"]  # Split quality measure
            # },
            # "XgBoost": {
            #     "n_estimators": [50, 100],
            #     "max_depth": [3, 5],
            #     "learning_rate": [0.01, 0.1]
            # }
        }
            from src.utils import evaluate_model
            model_report: dict = evaluate_model(X_train,y_train,X_test,y_test,models=models,param=param_grids)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model {best_model_name} found with accuracy of {best_model_score*100}")
            
            save_object(
                self.model_trainer_config.trainerd_model_path,
                best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_scorre = r2_score(y_test,predicted)
            
            return r2_scorre
        except Exception as e:
            raise CustomException(e,sys)
        