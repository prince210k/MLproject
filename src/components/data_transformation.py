import sys 
import os
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            num_cols = [
                "writing_score",
                "reading_score"
            ]
            cat_cols =[
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course'
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                    steps=[
                        ("imputer",SimpleImputer(strategy='most_frequent')),
                        ("OH_encoding",OneHotEncoder()),
                        ("Scaler",StandardScaler(with_mean=False))
                    ]
                )
            logging.info(f"Categorical Cols = {cat_cols}")
            
            logging.info(f"Numerical cols = {num_cols}")
            
            preprocessor = ColumnTransformer(
                [("Numerical_pipeline",num_pipeline,num_cols),
                ("Categorical_pipeline",cat_pipeline,cat_cols)]
            )
            
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Reading train and test completed")
            logging.info("Obtaining Preprocessor")
            
            preprocessing_obj = self.get_data_transformer_object()
            
            target_col = "math_score"
            num_cols = [
                        "writing_score",
                        "reading_score"
                    ]
            
            input_feature_train_df = train_df.drop(columns=[target_col],axis=1)
            target_feature_train_df = train_df[target_col]
            
            input_feature_test_df = test_df.drop(columns=[target_col],axis=1)
            target_feature_test_df = test_df[target_col]
            
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
            print(f"Transformed train features shape: {input_feature_train_arr.shape}")
            print(f"Train target shape: {target_feature_train_df.shape}")

            print(f"Transformed test features shape: {input_feature_test_arr.shape}")
            print(f"Test target shape: {target_feature_test_df.shape}")

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path = self.data_tranformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            