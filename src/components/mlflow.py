import optuna
import mlflow
import logging
from mlflow.tracking import MlflowClient

from src.constants import *
from src.logger import logging
from src.exception import CustomException
from src.config.configuration import *
from src.utils import *

import os, sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor


class Experiments_evaluation:
    def __init__(self, experiment_name, run_name):
        self.experiment_name = experiment_name
        self.run_name = run_name

        self.best_model_run_id = None
        self.best_model_uri= None
        self.model_path = None

        self.artifact_uri = None
        self.model_name = None

    def get_best_model_run_id(self, experiment_name, metric_name):
        # Get the experiment ID
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        # Retrieve runs and sort by the specific metric
        runs = mlflow.search_runs(experiment_ids=[experiment_id], filter_string='',
                                  order_by=[f"metrics.{metric_name} DESC"])
        
        if runs.empty:
            print("No runs found for the specified experiment and metric")
            return None
        
        # Get the ID of best run
        best_run = runs.iloc[0]
        self.best_model_run_id = best_run.run_id

        # Load the best mmodel
        self.best_model_uri = (f"runs:/{self.best_model_run_id}/model")

    def download_model(self, dst_path):
        self.artifact_uri = mlflow.get_run(self.best_model_run_id).info.artifact_uri
        model_uri = f"{self.artifact_uri}/{self.model_name}"
        model = mlflow.pyfunc.load_model(model_uri)
        save_object(file_path=dst_path, obj=model)


    def create_run_report(self):
        # Create MLflow client
        client = MlflowClient()
        run_id = self.best_model_run_id
        # Get run details
        run = client.get_run(run_id)

        # Report Data
        # List the contents of the artifact_uri directory
        model_name = self.model_name
        parameters = run.data.params
        metrics = str(run.data.metrics['R2_score']) # Retrive metrics

        return model_name, parameters, metrics
    
    def run_mlflow_experiment(self, R2_score, model, parameters, model_name):

        self.model_name = model_name
        # Create or get the experiment
        mlflow.set_experiment(self.experiment_name)

        # Start a run
        with mlflow.start_run(run_name=self.run_name):
            # Log metrics, param, and model
            mlflow.log_metric("R2_score",float(R2_score))
            mlflow.log_params(parameters)
            mlflow.sklearn.log_model(model, f"{model_name}")

        logging.info("Checking for best model from MLFLOW logs")

        self.get_best_model_run_id(metric_name='R2_score', experiment_name=self.experiment_name)

        print(f"Best model Run_ID:{self.best_model_run_id}")

        return self.best_model_run_id


class OptunaTuner:
    def __init__(self, model, params, X_train, y_train,X_test,y_test):
        self.model = model
        self.params = params
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def Objective(self, trial):
        param_values = {}
        for key, value_range in self.params.items():
            if value_range[0] <= value_range[1]:
                if isinstance(value_range[0], int) and isinstance(value_range[1], int):
                    param_values[key] = trial.suggest_int(key, value_range[0], value_range[1])
                else:
                    param_values[key] = trial.suggest_float(key, value_range[0], value_range[1])

            else:
                raise ValueError(f"Invalid value range for '{key}': low ={value_range}, hugh={value_range[1]}")
            
        self.model.set_params(**param_values)

        # Fit the model on the training data
        self.model.fit(self.X_train, self.y_train)

        # Predict on the test data
        y_pred = self.model.predict(self.X_test)

        # Calculate the R2_score as the objective is Maximize R2
        r2 = r2_score(self.y_test, y_pred)

        return r2
    
    def tune(self, n_trials=100):
        # Initialize the Tuner
        study = optuna.create_study(direction="maximize")
        study.optimize(self.Objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

        # Set the best parameters to the model
        self.model.set_params(**best_params)

        # Retrain the model with the best parameters on the entire training set
        self.model.fit(self.X_train, self.y_train)

        # Evaluate the model on the test set using R2_score on test set
        y_pred_test = self.model.predict(self.X_test)
        best_r2_score = r2_score(self.y_test, y_pred_test)
        print(f"Best R2 score on the Test Set"{best_r2_score})

        # Here we return the tuned model and the best R2 score on the test set
        return best_r2_score, self.model, best_params
    

    









@dataclass
class ModelTrainerConfig:
    trained_model_file_path = MODEL_FILE_PATH


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            X_train, y_train, X_test, y_test = (train_arr[:,:-1], train_arr[:,-1],
                                                test_arr[:,:-1], test_arr[:,-1])
            
            models = {
                "XGBRegressor": XGBRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "SVR": SVR()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            print(f"Best model found: {best_model}, R2_score: {best_model_score}")
            logging.info(f"Best model found: {best_model}, R2_score: {best_model_score}")

            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj= best_model)


        except Exception as e:
            raise CustomException(e,sys)



