import pandas as pd
import os


features = ["FGpct_diff", "FG3apct_diff", "FTpct_diff", "AST_diff", "OREB_diff", "DREB_diff", "STL_diff", "BLK_diff", "PF_diff", "PTS_diff"]
    
target = "local_wins"
#tratamiento de datos concretos y caracteristicas para el modelo final de prediccion de resultados de partidos

def GameID_Agrupation(gameLog_df):

    gameLog_groupby = gameLog_df.groupby("GAME_ID")
    rows = []

    for game_id, teams in gameLog_groupby:
        if len(teams) == 2:
            home = teams[teams["MATCHUP"].str.contains("vs").astype(int) == 1].iloc[0]
            away = teams[teams["MATCHUP"].str.contains("vs").astype(int) == 0].iloc[0]

            rows.append(
                {
                    "Game_ID": game_id,
                    "Local_team_id": home["Team_ID"],
                    "Visitor_team_id": away["Team_ID"],
                    "Visitor_WL": away["WL"],
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


def stats_comparation(gameLog_df):

    """
    convierte todas las estadisticas de las columnas numericas a valores absolutos
    ideal para calculos probabilisticos del modelo de correlacion lineal

    devuelve un dataFrame que contiene nada mas que los valores absolutos de las diferencias y los identificadores del partido
    """


    home_numeric_fields = ["Local_FgPct", "Local_FG3aPct", "Lolca_FtPct", "Local_AST", "Local_OREB", "Local_DREB", "Local_STL", "Local_BLK", "Local_PF", "Local_PTS"]
    away_numeric_fields = ["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct", "Visitor_AST", "Visitor_OREB", "Visitor_DREB", "Visitor_STL", "Visitor_BLK", "Visitor_PF", "Visitor_PTS"]
    new_abs_columns = ["FGpct_abs", "FG3apct_abs", "FTpct_abs", "AST_abs", "OREB_abs", "DREB_abs", "STL_abs", "BLK_abs", "PF_abs", "PTS_abs"]
    new_diff_columns = ["FGpct_diff", "FG3apct_diff", "FTpct_diff", "AST_diff", "OREB_diff", "DREB_diff", "STL_diff", "BLK_diff", "PF_diff", "PTS_diff"]

    gameLog_df[new_abs_columns] = abs(gameLog_df[home_numeric_fields].values - gameLog_df[away_numeric_fields].values) #calculo de competitividad de los valores
    gameLog_df[new_diff_columns] = gameLog_df[home_numeric_fields].values - gameLog_df[away_numeric_fields].values #calcuulo de analisis de la ventaja local sobre el visitante
    gameLog_df["local_wins"] = (gameLog_df["PTS_diff"] > 0).astype(int)

    
    

    return gameLog_df




def gameLog_feture_engineering(gameLog_df, proceded_path):

    """
    realiza la ingenieria de caracteristicas necesaria para el modelo de prediccion de resultados de partidos
    """

    gameLog_df = pd.read_csv(proceded_path)
    gameLog_df = GameID_Agrupation(gameLog_df)
    featured_gameLog_df = stats_comparation(gameLog_df)

    pd.to_csv(proceded_path, featured_gameLog_df, index = False)


    
    return features







