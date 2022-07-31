# -*- coding utf-8 -*-

# Data Science
import pandas as pd
import numpy as np
import yaml
import re
import pickle
import datetime


class PipelinePreprocessing(TransformerMixin):
    def __init__(self, model_config: dict):
        self.model_config = model_config

    def fit(self, df: pd.DataFrame):
        logger.info("[TBD] Fitting model...")


    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        logger.info("[ ] Starting transform...")
        model_config = self.model_config
        target_col = model_config["target_column_name"]
        
        # Get Cols
        if model_config['cols_to_keep']:
            features_df = features_df[
                model_config["cols_to_keep"] + [target_col]].drop_duplicates()


        features_df = features_df.drop_duplicates()

        # Target preprocessing
        features_df[target_col] = features_df[target_col].replace(model_config['target_feature_map']).astype(int)

        # Feature Cleaning

        # Feature Preprocessing

        if model_config["use_aggregation"]:
            features_df = self.aggregate_by_report(features_df)
        repeated_columns = [i for i in list(zip(*features_df.columns))[0] if
                            list(zip(*features_df.columns))[0].count(i) > 1]
        features_df.columns = [m + '/' + n if m in repeated_columns else m for m, n in features_df.columns]

        features_df.managment_level = features_df.managment_level.replace(
            model_config['management_level_map']).astype('int')

        logger.info("[X] Transform done")
        return features_df


def split_players_by_teams(players: pd.DataFrame, model_config):
    dire_data=players.groupby('match_id')['hero_id'].apply(list)
    radiant_data = pd.DataFrame()

    for j in range(dire_data.shape[0]):
        new_r=list()
        for i in range(5):
            new_r.append(dire_data[j].pop(dire_data[j].index(dire_data[j][0])))
        temp_dict={"Radiant_team":new_r}
        radiant_data=radiant_data.append(temp_dict,ignore_index=True) 

    return dire_data, radiant_data

def adding_mean_values_of_diff_features_by_team(df,groupby,features,new_df):
    for i in range(len(features)):
        dire_data=df.groupby(groupby)[features[i]].apply(list)
        radiant_data=pd.DataFrame()
        radiant_data=divide_by_team(dire_data,radiant_data)
        radiant_data=radiant_data['Radiant_team'].apply(lambda x : sum(x)/len(x))
        dire_data=dire_data.apply(lambda x :sum(x)/len(x))
        new_df[features[i]+'_radiant']=radiant_data
        new_df[features[i]+'_dire']=dire_data
    return new_df