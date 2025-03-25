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
            
            from src.utils import evaluate_model
            model_report: dict = evaluate_model(X_train,y_train,X_test,y_test,models=models)
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best model found with accuracy of {best_model_score}")
            
            save_object(
                self.model_trainer_config.trainerd_model_path,
                best_model
            )
            
            predicted = best_model.predict(X_test)
            
            r2_scorre = r2_score(y_test,predicted)
            
            return r2_scorre
        except Exception as e:
            raise CustomException(e,sys)
        