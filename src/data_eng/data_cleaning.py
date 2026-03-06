
import pandas as pd
import os




#modulo de limpieza de datos

#--------------------------------------FILTRADO DE DATOS------------------------------------------------------------------------------------------------

#filtrado de campos importantes de los partidos de hoy (fecha, id local, id visitante)
def gameDay_filter(rawPath, procesedPath):

    games_df = pd.read_csv(rawPath)

    campos_relevantes = ["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]

    games_df = games_df[campos_relevantes]

    games_df["GAME_DATE_EST"] = pd.to_datetime(games_df["GAME_DATE_EST"])

    procesed_csv = games_df.to_csv(procesedPath, index=False)
    return games_df


#filtrado de campos importantes de todos los partidos de gamelogs guardados()







def gameLog_csv_filter(rawPath, procesedPath):

    gameLog_df = pd.read_csv(rawPath)


    campos_relevantes = ["Team_ID", "Game_ID", "GAME_DATE","MATCHUP","WL","FGM","FGA","FG_PCT","FG3M","FG3_PCT","FTM", "FTA", "FT_PCT", "OREB", "DREB", "AST","STL","BLK","PF","PTS","TOV"]

    gameLog_df = gameLog_df[campos_relevantes]

    gameLog_df["GAME_DATE"] = pd.to_datetime(gameLog_df["GAME_DATE"])
    gameLog_df.columns = gameLog_df.columns.str.strip()
    campos_numericos = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "AST", "STL", "BLK", "PF", "PTS", "TOV"]

    #print(gameLog_df.columns.tolist())
    #print(campos_numericos)

    gameLog_df[campos_numericos] = gameLog_df[campos_numericos].apply(pd.to_numeric,errors = "coerce")
    gameLog_df = GameID_Agrupation(gameLog_df)
    #print("data frame despues de la agrupacion :", gameLog_df.head())
    gameLog_df = final_stats_validation(gameLog_df)
    #print("data frame despues de la validacion :", gameLog_df.head())

    final_csv = gameLog_df.to_csv(procesedPath,index=False)

    return gameLog_df





def final_stats_validation(gameLog_df):

    validated = gameLog_df.drop_duplicates(subset = ["Game_ID"])

    validated = validated [
        (validated["Local_PTS"] > 0) & (validated["Visitor_PTS"] > 0) 
    ]

    return validated





#tratamiento de datos del historial de rendimiento por equipo por temporada


def TeamStatistics_csv_filter(rawPath, procesedPath):

    treatment_statistics_df = pd.read_csv(rawPath)

    campos_iniciales = ["TEAM_ID","TEAM_NAME","W","L","W_PTC","FGA","FTA","OREB","DREB",]
    campos_numericos = ["FGA","FTA","OREB","DREB","W","L","W_PTC"]

    final_df = treatment_statistics_df[campos_iniciales]
    final_df[campos_numericos] = final_df[campos_numericos].apply(pd.to_numeric,errors = "coerce")
    final_df = final_df.drop_duplicates(subset = ["Team_ID"])

    final_csv = final_df.to_csv(procesedPath,index=False)
    
    return final_df



def GameID_Agrupation(gameLog_df):

    is_home = gameLog_df["MATCHUP"].str.contains("vs")
    home_df = gameLog_df[is_home].copy()
    away_df = gameLog_df[~is_home].copy()

    home_cols = {col: "Local_" + col for col in gameLog_df.columns if col not in ["Game_ID", "GAME_DATE"]}
    away_cols = {col: "Visitor_" + col for col in gameLog_df.columns if col not in ["Game_ID", "GAME_DATE"]}

    home_df = home_df.rename(columns=home_cols)
    away_df = away_df.rename(columns=away_cols)

    merged_df = pd.merge(home_df, away_df, on=["Game_ID", "GAME_DATE"], how="inner")

    rename_mapping = {
        "Local_Team_ID": "Local_team_id",
        "Visitor_Team_ID": "Visitor_team_id",
        "Local_FG_PCT": "Local_FgPct",
        "Visitor_FG_PCT": "Visitor_FgPct",
        "Local_FG3_PCT": "Local_FG3aPct",
        "Visitor_FG3_PCT": "Visitor_FG3aPct",
        "Local_FT_PCT": "Local_FtPct",
        "Visitor_FT_PCT": "Visitor_FtPct",
    }
    final_gameLogs_df = merged_df.rename(columns=rename_mapping)
    return final_gameLogs_df
