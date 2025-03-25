## For Reading data from sources
import os 
import sys
import pandas as pd 
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig 

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig():
    train_data_path: str  = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')


class DataIngestion():
    def __init__(self):
        self.config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        try:
            logging.info("Starting Data Ingestion Process...")
            data_path = os.path.join('notebook/data/stud.csv')
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")
            df = pd.read_csv(data_path)
        
            logging.info(f"Data Ingestion Completed with shape {df.shape}")
            
            os.makedirs(os.path.dirname(self.config.raw_data_path),exist_ok=True)
            df.to_csv(self.config.raw_data_path,index=False)
            
            logging.info("Performing Train Test Split")
            
            train_data,test_data = train_test_split(df,test_size=0.2,random_state=42)
            
            train_data.to_csv(self.config.train_data_path,index=False,header=True)
            test_data.to_csv( self.config.test_data_path,index=False,header=True)
            
            logging.info("data ingestion completed")
            return self.config.train_data_path,self.config.test_data_path
        
        except Exception as e:
            logging.error(f"Error in Data Ingestion Process {str(e)}")
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    