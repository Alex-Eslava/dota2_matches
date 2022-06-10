import sys
import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[3]

sys.path.append(str(PROJECT_DIR))

LOAD_FROM_LOCAL = True
OUTLIERS_MODEL_PATH = 'models/outliers_default_202203231033'  # 'models/outliers_default_202203221119'
CATBOOST_MODEL_PATH = 'models/catboost_default_202203221159'





if __name__ == "__main__":
    print('Starting forecast')


    # 1. Get data
    print("Reading data...")
    ## Ideally here we should get the data from the Dota2 API not csv files
    data_prefix = '../data/external/dota2_dataset'
    players=pd.read_csv(f'{data_prefix}/players.csv')
    matches=pd.read_csv(f'{data_prefix}/match.csv')
    teamfights=pd.read_csv(f'{data_prefix}/teamfights.csv')

    # 2. Prepare data
    df_actual_out = df_actual[['report_id', 'approval_status']]
    df_actual_out['type'] = 'actual'
    # 3. Predict
    model = CatboostClassifier.from_local(CATBOOST_MODEL_PATH)

    print('Forecasting...')
    pred = model.predict(to_predict)
    pred = pred.reset_index()
    df_ai_out = pred
    df_ai_out.columns = ['report_id', 'approval_status']
    df_ai_out['type'] = 'ai'


    # 7. Prepare output
    out = df_ai_out

    print(out)
