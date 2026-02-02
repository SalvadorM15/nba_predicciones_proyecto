from data_eng import data_collection as dc
from data_eng import data_cleaning as dCl
import pandas as pd
from data_an import feature_eng as eng
import os

seasons = [
    "2024-25"
]
gameLog_df = dc.fetch_teams_gameLogs(seasons)
clean_gameLog_df = dCl.gameLog_csv_filter("data/raw_data/all_teams_gameLogs.csv","data/procesed_data/gameLogs_clean.csv")
featured_gameLog_df = eng.GameLog_features(20,"data/procesed_data/gameLogs_clean.csv",eng.Get_Id_by_Name("Boston Celtics"),eng.Get_Id_by_Name("Miami Heat"))

#stats_df = dc.fetch_teams_statistics(seasons)

#gameDays_df = dc.fetch_gameDays()
#clean_gameDays_df = dCl.gameDay_filter("data/raw_data/gameDays.csv","data/procesed_data/gameDays_clean.csv")


print(featured_gameLog_df.head())