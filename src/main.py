import data_collection as dc
import data_cleaning as dCl
import os

seasons = [
    "2024-25"
]
gameLog_df = dc.fetch_teams_gameLogs(seasons)
clean_gameLog_df = dCl.gameLog_csv_filter("data/raw_data/all_teams_gameLogs.csv","data/procesed_data/gameLogs_clean.csv")
#stats_df = dc.fetch_teams_statistics(seasons)

#gameDays_df = dc.fetch_gameDays()
#clean_gameDays_df = dCl.gameDay_filter("data/raw_data/gameDays.csv","data/procesed_data/gameDays_clean.csv")


print(clean_gameLog_df.head())