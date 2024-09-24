import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import os,sys
from src.config.configuration import *
from dataclasses import dataclass


# Class - Feature Engineering & Data Transformation

class Feature_Engineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        logging.info('*** Feature_Engineering Starts ***')
    
    def distance_numpy(self, df, lat1, lat2, lon1, lon2):
        p = np.pi/100
        a = 0.5 - np.cos((df[lat2]-df[lat1])*p)/2 + np.cos(df[lat1]*p) * np.cos(df[lat2]*p) * (1-np.cos((df[lon2])))
        df['distance'] = 12734 * np.arccos(np.sort(a))

    def transform_data(self, df):
        try:
            self.distance_numpy(df,'Restaurant_latitude','Restaurant_longitude','Delivery_location_latitude','Delivery_location_longitude')

            df.drop(['ID','Restaurant_latitude','Restaurant_longitude',
                     'Delivery_location_latitude','Delivery_location_longitude',
                     'Delivery_person_ID','Order_Date','Time_Orderd','Time_Order_picked'], axis=1, inplace=True)
            
            logging.info('Dropping Columns from Original Dataset')

            return df

        except Exception as e:
            raise CustomException(e,sys)
        
    def fit(self, X, y=None):
        return self
  
    def transform(self, X:pd.DataFrame, y=None):
        try:
            transformed_df = self.transform_data(X)
            return transformed_df
        except Exception as e:
            raise CustomException(e,sys)

@dataclass
class DataTransformationConfig():
    feature_engg_obj_path = FEATURE_ENGG_OBJ_PATH
    processed_obj_file_path = PREPROCESSING_OBJ_FILE
    transformed_train_path = TRANSFORMED_TRAIN_FILE_PATH
    transformed_test_path = TRANSFORMED_TEST_FILE_PATH

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformation_obj(self):
        try:
            Road_traffic_density = ['Low','Medium','High','Jam']
            Weather_conditions = ['Sunny','Cloudy','Windy','Fog','Sandstorms','Stormy']

            categorical_column = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
            ordinal_column = ['Road_traffic_density','Weather_conditions']
            numerical_column = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
                                'multiple_deliveries','distance']

            numerical_pipeline = Pipeline(steps=[
                ('impute',SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
                ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])
            ordinal_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density,Weather_conditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])
            
            preprocessor = ColumnTransformer([
                    ('num_pipeline', numerical_pipeline, numerical_column),
                    ('cat_pipeline', categorical_pipeline, categorical_column),
                    ('ord_pipeline', ordinal_pipeline, ordinal_column)
                ]
            )
            save_object(file_path= self.data_transformation_config.processed_obj_file_path,obj=preprocessor)
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        

    def get_feature_engg_obj(self):
        try:
            feature_engineering = Pipeline(steps = [("fe", Feature_Engineering())])
            save_object(file_path=self.data_transformation_config.feature_engg_obj_path, obj=feature_engineering)
            return feature_engineering
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Loading train & test data for preprocessing & feature engineering")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Data Loaded")

            logging.info("Loading Feature Engineering Object")
            fe_obj = self.get_feature_engg_obj()
            logging.info("Applying Feature Engineering")
            train_df = fe_obj.fit_transform(train_df)
            test_df = fe_obj.transform(test_df)


            logging.info("Separating Target variables from train & test data")
            X_train = train_df.drop(['Time_taken (min)'], axis=1)
            y_train = train_df['Time_taken (min)']

            X_test = test_df.drop('Time_taken (min)', axis=1)
            y_test = test_df['Time_taken (min)']

            logging.info(f"Shape of X_train:{X_train.shape}\nShape of X_test:{X_test.shape}\nShape of y_train:{y_train.shape}\nShape of y_test:{y_test.shape}")

            processing_obj = self.get_data_transformation_obj()
            logging.info("Applying preprocessing object to X_train & X_test")
            X_train = processing_obj.fit_transform(X_train)
            X_test = processing_obj.transform(X_test)
            logging.info("Data Preprocessed Sucessfully")

            train_arr = np.c_[X_train, np.array(y_train)]
            test_arr = np.c_[X_test, np.array(y_test)]
            logging.info("Target variable added back to train & test arrays")

            df_train = pd.DataFrame(train_arr)
            df_test = pd.DataFrame(test_arr)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_path), exist_ok=True)
            df_train.to_csv(self.data_transformation_config.transformed_train_path, index=False)

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_test_path), exist_ok=True)
            df_test.to_csv(self.data_transformation_config.transformed_test_path, index=False)

            # save_object(self.data_transformation_config.processed_obj_file_path, obj=fe_obj)
            
            # save_object(self.data_transformation_config.feature_engg_obj_path, obj=fe_obj)


            return (
                train_arr,
                test_arr,
                # self.data_transformation_config.processed_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)