
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


    campos_relevantes = ["Team_ID", "Game_ID", "GAME_DATE","MATCHUP","WL","FGM","FGA","FG_PCT","FG3M","FG3_PCT","FTM", "FTA", "FT_PCT", "OREB", "DREB", "AST","STL","BLK","PF","PTS"]

    gameLog_df = gameLog_df[campos_relevantes]

    gameLog_df["GAME_DATE"] = pd.to_datetime(gameLog_df["GAME_DATE"])
    print(gameLog_df.columns.tolist())
    gameLog_df.columns = gameLog_df.columns.str.strip()
    campos_numericos = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB", "DREB", "AST", "STL", "BLK", "PF", "PTS"]
    gameLog_df[campos_numericos] = gameLog_df[campos_numericos].apply(pd.to_numeric,errors = "coerce")

    gameLog_df = gamelog_purification(gameLog_df)
    gameLog_df = final_stats_validation(gameLog_df)

    final_csv = gameLog_df.to_csv(procesedPath,index=False)

    return gameLog_df










def gamelog_purification (original_gameLog):

    #buscamos crear un nuevo dataframe purificado a partir de las columnas que seleccionamos
    #la funcion va a devolver un dataFrame el cual sera el que queda persistido y con el cual trabajaremos en la seccion de feature_eng
    
    """
    gamelog final structure:
    
        gamelog_df (

            Local_team_id,
            visitor_team_id,
            game_id,
            visitor_scoring_stats,
            local_scoring_stats,
            visitor_defensive_stats,
            local_defensive_stats,
        )
    """

    gameLog_groupby = original_gameLog.groupby("Game_ID")
    rows = []

    for game_id, teams in gameLog_groupby:
        if len(teams) == 2:
            home = teams[teams["MATCHUP"].str.contains("vs").astype(int) == 1].iloc[0]
            away = teams[teams["MATCHUP"].str.contains("vs").astype(int) == 0].iloc[0]

            rows.append(
                {
                    "Game_id": game_id,
                    "Local_team_id": home["Team_ID"],
                    "Visitor_team_id": away["Team_ID"],
                    "Visitor_WL": away["WL"],\
                    "Local_WL": home["WL"],
                    "Visitor_FgPct": away["FG_PCT"],
                    "Local_FgPct": home["FG_PCT"],
                    "Visitor_FG3aPct": away["FG3_PCT"],
                    "Local_FG3aPct": home["FG3_PCT"],
                    "Visitor_FtPct": away["FT_PCT"],
                    "Local_FtPct": home["FT_PCT"],
                    "Visitor_AST": away["AST"],
                    "Local_AST": home["AST"],
                    "Visitor_OREB": away["OREB"],
                    "Local_OREB": home["OREB"],
                    "Visitor_DREB": away["DREB"],
                    "Local_DREB": home["DREB"],
                    "Visitor_STL": away["STL"],
                    "Local_STL": home["STL"],
                    "Visitor_BLK": away["BLK"],
                    "Local_BLK": home["BLK"],
                    "Visitor_PF": away["PF"],
                    "Local_PF": home["PF"],
                    "Visitor_PTS": away["PTS"],
                    "Local_PTS": home["PTS"],
                }
            )


    final_gameLogs_df = pd.DataFrame(rows)
    return final_gameLogs_df
    #


def final_stats_validation(gameLog_df):

    validated = gameLog_df.drop_duplicates(subset = ["Game_ID"])

    validated = validated [
        (validated["Local_PTS"] > 0) & (validated["Visitor_PTS"] > 0) 
    ]

    return validated





#tratamiento de datos del historial de rendimiento por equipo por temporada


def TeamStatistics_csv_filter(rawPath, procesedPath):

    treatment_statistics_df = pd.read_csv(rawPath)

    campos_iniciales = ["TEAM_ID","TEAM_NAME","W","L","W_PTC","FGA","FTA","OREB",]
    