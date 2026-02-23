from nba_api.stats.endpoints import teamgamelog, leaguedashteamstats, scoreboardv2
from nba_api.stats.static import teams
import pandas as pd
import time
import os




#---------------modulos de recopilacion de datos via nba_api------------------------------------------------------------\

#recopilacion datos de equipos

#"fetch_teams_gamelogs" trae los resultados y algunas estadisticas de todos los partidos de las temporadas que vengan en la coleccion "seasons"

def fetch_teams_gameLogs(seasons):

    all_teams = teams.get_teams()
    all_data = []

    for season in seasons:

        for team in all_teams:

            team_id = team["id"]
            team_name = team["full_name"]

            try:
                gamelog = teamgamelog.TeamGameLog(
                    team_id = team_id,
                    season = season 
                )


            #creo el data frame

                df = gamelog.get_data_frames()[0]
                df["SEASON"] = season

                all_data.append(df)

                time.sleep(0.8)

            except Exception as e:
                print("error en el equipo: {team_name}: {e}")

    final_df = pd.concat(all_data, ignore_index= True)

    file_path = "../data/raw_data/all_teams_gamelogs.csv"
    final_df.to_csv(file_path, index= False)

    return final_df

#recopilacion de estadisticas de equipos


# "fetch_teams_statistics" trae el rendimiento de cada equipo de la nba durante una determinada cantidad de temporadas sabendo su record
def fetch_teams_statistics(seasons):

    all_data = []

    for season in seasons:
        try:
            stats = leaguedashteamstats.LeagueDashTeamStats(
                 season = season,
                 season_type_all_star= "Regular Season"
            )

            df = stats.get_data_frames()[0]

            all_data.append(df)
            time.sleep(0.8)

        except Exception as e:
            print("error {e}")



    final_df = pd.concat(all_data, ignore_index= True)
    file_path = "data/raw_data/all_teams_statistics.csv"    
    final_df.to_csv(file_path, index = False)

    return final_df



#proximamente recopilacion de mas informacion de jugadores

def fetch_gameDays():
    
    all_data = []

    try:

        game_days = scoreboardv2.ScoreboardV2()

        df = game_days.get_data_frames()[0]

        all_data.append(df)
        
    except Exception as e:
        print("ocurrio un error en la lectura de calendario de partidos: {e}")

    final_df = pd.concat(all_data,ignore_index=True)
    file_path = "data/raw_data/gameDays.csv"

    final_df.to_csv(file_path,index=False)

    return final_df 
