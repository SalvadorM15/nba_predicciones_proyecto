from nba_api.stats.endpoints import teamgamelog, leaguedashteamstats, scoreboardv2, leaguegamelog
from nba_api.stats.static import teams
import pandas as pd
import time
import os
from datetime import datetime




#---------------modulos de recopilacion de datos via nba_api------------------------------------------------------------\

#recopilacion datos de equipos

#"fetch_teams_gamelogs" trae los resultados y algunas estadisticas de todos los partidos de las temporadas que vengan en la coleccion "seasons"

def fetch_teams_gameLogs(seasons):

    all_data = []

    for season in seasons:
        try:
            gamelog = leaguegamelog.LeagueGameLog(season=season)
            df = gamelog.get_data_frames()[0]
            df.rename(columns={"TEAM_ID": "Team_ID", "GAME_ID": "Game_ID"}, inplace=True)
            df["SEASON"] = season

            all_data.append(df)
            time.sleep(1)

        except Exception as e:
            print(f"error obteniendo gamelogs de la season {season}: {e}")

    final_df = pd.concat(all_data, ignore_index= True)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(project_root, "data", "raw_data", "all_teams_gamelogs.csv")
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
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(project_root, "data", "raw_data", "all_teams_statistics.csv")    
    final_df.to_csv(file_path, index = False)

    return final_df



def fetch_gameDays():
    all_data = []

    try:
        game_days = scoreboardv2.ScoreboardV2()
        df = game_days.get_data_frames()[0]
        all_data.append(df)
        
    except Exception as e:
        print(f"ocurrio un error en la lectura de calendario de partidos: {e}")

    final_df = pd.concat(all_data,ignore_index=True)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    file_path = os.path.join(project_root, "data", "raw_data", "gameDays.csv")

    final_df.to_csv(file_path,index=False)

    return final_df 

def fetch_incremental_gamelogs(latest_date, season="2024-25"):
    """
    Descarga los partidos de todos los equipos y filtra solo los que
    ocurrieron estrictamente después de latest_date. Luego los añade al CSV principal.
    Devuelve los deltas filtrados crudos.
    """
    try:
        gamelog = leaguegamelog.LeagueGameLog(season=season)
        full_updated_df = gamelog.get_data_frames()[0]
        full_updated_df.rename(columns={"TEAM_ID": "Team_ID", "GAME_ID": "Game_ID"}, inplace=True)
        full_updated_df["SEASON"] = season
    except Exception as e:
        print(f"Error extrayendo datos incrementales rápidos: {e}")
        return None

    full_updated_df["GAME_DATE"] = pd.to_datetime(full_updated_df["GAME_DATE"])
    
    # Filtrar estrictamente fechas nuevas
    new_games_df = full_updated_df[full_updated_df["GAME_DATE"] > pd.to_datetime(latest_date)]
    
    if new_games_df.empty:
        return new_games_df
        
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    raw_path = os.path.join(project_root, "data", "raw_data", "all_teams_gamelogs.csv")
    
    # Sobreescribimos el CSV global con toda la data junta para mantener el "master data" fresco.
    # Dado que la API igual devuelve toda la season de cada equipo (no permite limite custom por fecha)
    # guardamos todo, pero solo retornamos las filas nuevas para Features.
    # Evitamos duplicados guardando al disco.
    full_updated_df.to_csv(raw_path, index=False)
    
    return new_games_df

def get_last_seasons(n=5):
    """
    Retorna una lista con las últimas 'n' temporadas.
    El formato devuelto es el que usa la API de NBA (ej: '2024-25').
    """
    now = datetime.now()
    # Si estamos en octubre o después, la temporada actual empezó en base al año actual
    if now.month >= 10:
        current_start_year = now.year
    else:
        current_start_year = now.year - 1
        
    seasons = []
    # Genera n temporadas hasta la actual
    for i in range(n - 1, -1, -1):
        start_year = current_start_year - i
        end_year_str = str(start_year + 1)[-2:]
        seasons.append(f"{start_year}-{end_year_str}")
        
    return seasons
