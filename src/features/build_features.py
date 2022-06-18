# -*- coding utf-8 -*-

# Data Science
import pandas as pd
import numpy as np
import yaml
import re
import pickle
import datetime


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