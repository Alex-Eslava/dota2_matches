
import numpy as np
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

    return grid_cv_fitted.best_params_

def get_model_with_config(model_config, train_data):
    params = {}

    if model_config['type']=='catboost':
        categorical_features_indices = np.where(train_data.dtypes != float)[0]
        if model_config['free_text_cols']:
            text_features_indices = np.where(train_data.columns in model_config['free_text_cols'])[0]
        params["text_features"] = text_features_indices
        params["text_processing"] = [
            'NaiveBayes+Word|BoW+Word,BiGram|BM25+Word'
        ]
        model = CatBoostClassifier(**params, cat_features=categorical_features_indices)
    else:
        print("errroooor, throw exception here lad")    

    return model