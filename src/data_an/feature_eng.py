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

        """


        # creo las listas con los campos que seran necesarios para los futuros calculos
        home_numeric_fields = ["Local_FgPct", "Local_FG3aPct", "Local_FtPct", "Local_AST", "Local_OREB", "Local_DREB", "Local_STL", "Local_BLK", "Local_PF", "Local_PTS"]
        away_numeric_fields = ["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct", "Visitor_AST", "Visitor_OREB", "Visitor_DREB", "Visitor_STL", "Visitor_BLK", "Visitor_PF", "Visitor_PTS"]
        features_numeric_fields = ["FGpct_diff", "FG3apct_diff", "FTpct_diff", "AST_diff", "OREB_diff", "DREB_diff", "STL_diff", "BLK_diff", "PF_diff", "PTS_diff"]
        
        gameLog_df = pd.read_csv(proceded_path, index_col=False)


        #creo el data frame de los equipos que quiero buscar con una cola de n partidos

        home_team_df = gameLog_df[gameLog_df["Local_team_id"] == home_team_id].sort_values(by="Game_ID").tail(n)[home_numeric_fields]
        away_team_df = gameLog_df[gameLog_df["Visitor_team_id"] == away_team_id].sort_values(by="Game_ID").tail(n)[away_numeric_fields]

        #calculo los valores del data frame final que seran los features que usare
    
        features = pd.DataFrame()
        features[features_numeric_fields] = home_team_df.mean().values - away_team_df.mean().values
        

        return features



