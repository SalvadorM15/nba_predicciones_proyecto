import pandas as pd
import os


features = ["FGpct_diff", "FG3apct_diff", "FTpct_diff", "AST_diff", "OREB_diff", "DREB_diff", "STL_diff", "BLK_diff", "PF_diff", "PTS_diff"]
    
target = "local_wins"
#tratamiento de datos concretos y caracteristicas para el modelo final de prediccion de resultados de partidos





def gameLog_feture_engineering(gameLog_df, proceded_path):

    """
    realiza la ingenieria de caracteristicas necesaria para el modelo de prediccion de resultados de partidos
    """

    gameLog_df = pd.read_csv(proceded_path)
    gameLog_df = GameID_Agrupation(gameLog_df)
    featured_gameLog_df = stats_comparation(gameLog_df)

    pd.to_csv(proceded_path, featured_gameLog_df, index = False)


    
    return features


def Get_Id_by_Name(team_name):

    """
    obtiene el id del equipo a partir de su nombre
    """

    teams_dict = {
        "Atlanta Hawks": 1610612737,
        "Boston Celtics": 1610612738,
        "Brooklyn Nets": 1610612751,
        "Charlotte Hornets": 1610612766,
        "Chicago Bulls": 1610612741,
        "Cleveland Cavaliers": 1610612739,
        "Dallas Mavericks": 1610612742,
        "Denver Nuggets": 1610612743,
        "Detroit Pistons": 1610612765,
        "Golden State Warriors": 1610612744,
        "Houston Rockets": 1610612745,
        "Indiana Pacers": 1610612754,
        "Los Angeles Clippers": 1610612746,
        "Los Angeles Lakers": 1610612747,
        "Memphis Grizzlies": 1610612763,
        "Miami Heat": 1610612748,
        "Milwaukee Bucks": 1610612749,
        "Minnesota Timberwolves": 1610612750,
        "New Orleans Pelicans": 1610612740,
        "New York Knicks": 1610612752,
        "Oklahoma City Thunder": 1610612760,
        "Orlando Magic": 1610612753,
        "Philadelphia 76ers": 1610612755,
        "Phoenix Suns": 1610612756,
        "Portland Trail Blazers": 1610612757,
        "Sacramento Kings": 1610612758,
        "San Antonio Spurs": 1610612759,
        "Toronto Raptors": 1610612761,
        "Utah Jazz": 1610612762,
        "Washington Wizards": 1610612764
    }

    return teams_dict.get(team_name, None)






#funcion principal para obtener los features del historial de partidos
def GameLog_features (n, proceded_path, home_team_id, away_team_id):
    
        """
        obtiene el rendimiento en anotaciones, resultados y defensa de ambos equipos en los ultimos n partidos
        luego calcula las diferencias entre ambos equipos en cada estadistica relevante
        y devuelve una lista con los features calculados

        datos crudos calculados:
            ..._pct : todos los porcentajes;
        
        datos relativos:

            AST, STL,BLK,PTS: asistencias, robos y tapones calculados por la cantidad de poseciones del equipo o del equipo contrario
            OREB, DREB: rebotes ofensivos y defensivos calculados por la cantidad de tiros errados del equipo o del equipo contrario

        """


        # creo las listas con los campos que seran necesarios para los futuros calculos
        home_numeric_fields = ["Local_FgPct", "Local_FG3aPct", "Local_FtPct", "Local_AST_rate", "Local_OREB_rate", "Local_DREB_rate", "Local_STL_rate", "Local_BLK_rate", "Local_PF_rate", "Local_PTS_rate"]
        away_numeric_fields = ["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct", "Visitor_AST_rate", "Visitor_OREB_rate", "Visitor_DREB_rate", "Visitor_STL_rate", "Visitor_BLK_rate", "Visitor_PF_rate", "Visitor_PTS_rate"]
        diff_numeric_fields = ["FGpct_diff", "FG3apct_diff", "FTpct_diff",]
        rate_numeric_fields = ["AST_rate_diff", "OREB_rate_diff", "DREB_rate_diff", "STL_rate_diff", "BLK_rate_diff", "PF_rate_diff", "PTS_rate_diff"]
        
        gameLog_df = pd.read_csv(proceded_path, index_col=False)

        #calculo de las posiciones aproximadas de un equipo

        gameLog_df["Local_POSS"] = gameLog_df["Local_FGA"] + 0.4 * gameLog_df["Local_FTA"] - 1.07 * (gameLog_df["Local_OREB"] / (gameLog_df["Local_OREB"] + gameLog_df["Visitor_DREB"])) * (gameLog_df["Local_FGA"] - gameLog_df["Local_FGM"]) + gameLog_df["Local_TOV"]
        gameLog_df["Visitor_POSS"] = gameLog_df["Visitor_FGA"] + 0.4 * gameLog_df["Visitor_FTA"] - 1.07 * (gameLog_df["Visitor_OREB"] / (gameLog_df["Visitor_OREB"] + gameLog_df["Local_DREB"])) * (gameLog_df["Visitor_FGA"] - gameLog_df["Visitor_FGM"]) + gameLog_df["Visitor_TOV"]

        #calculo la cantidad de tiros fallados por cada equipo
        gameLog_df["Local_FGMissed"] = gameLog_df["Local_FGA"] - gameLog_df["Local_FGM"]
        gameLog_df["Visitor_FGMissed"] = gameLog_df["Visitor_FGA"] - gameLog_df["Visitor_FGM"]


        #una vez teniendo los datos necesarios calculo los rates
        gameLog_df = rate_calculator(gameLog_df, ["Local_AST","Local_PTS"], "Local_POSS")
        gameLog_df = rate_calculator(gameLog_df, ["Visitor_BLK","Visitor_STL"], "Visitor_POSS")
        gameLog_df = rate_calculator(gameLog_df, ["Local_DREB"], "Visitor_FGMissed")
        gameLog_df = rate_calculator(gameLog_df, ["Visitor_OREB"], "Local_FGMissed")
        
        #selecciono los ultimos n partidos de cada equipo
        home_team_df = gameLog_df[gameLog_df["Local_team_id"] == home_team_id].sort_values(by="Game_ID").tail(n)[home_numeric_fields]
        away_team_df = gameLog_df[gameLog_df["Visitor_team_id"] == away_team_id].sort_values(by="Game_ID").tail(n)[away_numeric_fields]

        
        home_team_df= home_team_df.sum().to_frame().T
        away_team_df = away_team_df.sum().to_frame().T

        #calculo los valores del data frame final que seran los features que usare
    
        features = pd.DataFrame(columns = diff_numeric_fields + rate_numeric_fields)
        features[diff_numeric_fields] = (

        home_team_df[["Local_FgPct", "Local_FG3aPct", "Local_FtPct"]].values -
        away_team_df[["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct"]].values
        
        )
        features[rate_numeric_fields] =(

        home_team_df[["Local_AST_rate", "Local_OREB_rate", "Local_DREB_rate", "Local_STL_rate", "Local_BLK_rate", "Local_PF_rate", "Local_PTS_rate"]].values - 
        away_team_df [["Visitor_AST_rate", "Visitor_OREB_rate", "Visitor_DREB_rate", "Visitor_STL_rate", "Visitor_BLK_rate", "Visitor_PF_rate", "Visitor_PTS_rate"]].values  

        )
        return features





def rate_calculator(gameLog_df, stats_cols : list, reference : str) -> pd.DataFrame:

    """
        calcula el atributo rate para cualquier estadistica pasada en stats_cols en relacion al parametro reference

        reference puede ser "Local_POSS" o "Visitor_POSS" dependiendo si se quiere calcular el rate para el equipo local o visitante
        tambien puede ser "Visitor_FGMissed" o "Local_FGMissed" si se quiere calcular el rate en relacion a los tiros fallados del equipo contrario

        devuelve el data frame con las nuevas columnas calculadas
    """

    for stat in stats_cols:
        gameLog_df[stat + "_rate"] = gameLog_df[stat] / gameLog_df[reference]


    return gameLog_df
