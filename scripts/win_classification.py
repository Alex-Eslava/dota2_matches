import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np


from dota2_matches.src.features import build_features as bf

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
    dire, radiant = bf.split_players_by_teams(players=players)
    matches['Radiant_team']=radiant
    matches['Dire_team']=dire
    matches=matches.drop(columns=['start_time','duration','game_mode','positive_votes','negative_votes','cluster'])

    features=['gold_spent','gold_per_min','xp_per_min','kills','deaths','tower_damage','hero_damage']
    matches=bf.adding_mean_values_of_diff_features_by_team(
        df=players, 
        groupby='match_id', 
        features=features, 
        new_df = matches)

    # Store this <--- tbd

    x_train, x_test, y_train,y_test = split_scale(matches, 'radiant_win')

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
