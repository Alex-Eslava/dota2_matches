
import numpy as np
import pandas as pd

import argparse
import pickle
import sys
import time
import yaml


from datetime import datetime
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier


from dota2_matches.lib_d2c import utils



def grid_search(model_config, train_data):
    print(f"Gridsearch starting with config: {model_config['config_name']}")

    target_label = model_config['target_column_name']

    # Only train data and val data are going to be used in the Hyperparameter tuning task
    preprocessing_pipeline = PipelinePreprocessing(model_config)
    preprocessing_pipeline.fit(train_data)

    train_data = prepare_for_train(train_data)
       
    train_data, val_data = train_test_split(train_data, test_size=model_config['val_size'])

    train_x, train_y, val_x, val_y = utils.split_target_column(train_data=train_data, test_data=val_data, target=target_label)

    # Parameter tuning
    param_tuning_grid = model_config["param_tuning_grid"]
    model = get_model_with_config(model_config=model_config, train_data=train_x)

    grid_cv = GridSearchCV(
        model,
        param_tuning_grid,
        random_state=model_config["random_state"],
        **2["param_tuning_config"]
    )

    grid_cv_fitted = grid_cv.fit(
        train_x,
        train_y,
        eval_set=(val_x, val_y),
        early_stopping_rounds=42
    )

    return grid_cv_fitted, grid_cv_fitted.best_params_

def get_model_with_config(model_config, train_data):
    params = {}

    if model_config['type'].lower()=='catboost':
        categorical_features_indices = np.where(train_data.dtypes != float)[0]
        if model_config['free_text_cols']:
            text_features_indices = np.where(train_data.columns in model_config['free_text_cols'])[0]
        params["text_features"] = text_features_indices
        params["text_processing"] = [
            'NaiveBayes+Word|BoW+Word,BiGram|BM25+Word'
        ]
        model = CatBoostClassifier(**params, cat_features=categorical_features_indices)
    elif model_config['type'].lower()=='logisticregression':
        # Simple model to test out the waters 
        try:
            from sklearn.linear_model import LogisticRegressionCV
        except ModuleNotFoundError as e:
            print("My man get your pip updated")
        model = LogisticRegressionCV()
    else:
        print("unsupported, we're working on it")    

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--train-best-model', action='store_true', default=False)
    parser.add_argument('--test-best-model', action='store_true', default=False)
    parser.add_argument('--grid-search', action='store_true', default=False)

    args = parser.parse_args()

    start_time = time.time()
    config_file = "config/model_configs/model_win_predict.yaml"

    model_config = yaml.safe_load(open(config_file))

    print("Reading data")
    raw_data_path = "data/interim/merged_data.csv"
    base_data = 

    if args.grid_search:
        print("Grid search on model parameters")
        model, grid_search_metrics = grid_search(model_config, base_data)

    if args.test_best_model:
        print("Testing best model")
        test_set_metrics = test_best_model(model_config, base_data)
        print(test_set_metrics)

    if args.train_best_model:
        print("Training best model")
        m = train_best_model("models", model_config, base_data)
        print("Model saved!")

    print("--- %s seconds ---" % (time.time() - start_time))
