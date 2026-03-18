import pandas as pd
import numpy as np
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




def _safe_divide(numerator, denominator):
    """Division segura que reemplaza inf/NaN por 0."""
    result = numerator / denominator.replace(0, np.nan)
    return result.fillna(0).replace([np.inf, -np.inf], 0)


def _get_season_from_date(game_date):
    """Determina la temporada NBA a partir de una fecha (ej: 2024-11-15 -> '2024-25')."""
    if isinstance(game_date, str):
        game_date = pd.to_datetime(game_date)
    year = game_date.year
    month = game_date.month
    if month >= 10:
        start_year = year
    else:
        start_year = year - 1
    end_year_str = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_str}"


def _get_standings_features(home_team_id, away_team_id, game_date, standings_df):
    """
    Calcula features de standings: diferencial de WinPCT y Rank entre local y visitante.
    Retorna (Standings_WinPct_diff, Standings_Rank_diff) o (0, 0) si no hay datos.
    """
    if standings_df is None or standings_df.empty:
        return 0.0, 0.0
    
    season = _get_season_from_date(game_date)
    
    home_standing = standings_df[(standings_df["TeamID"] == home_team_id) & (standings_df["SEASON"] == season)]
    away_standing = standings_df[(standings_df["TeamID"] == away_team_id) & (standings_df["SEASON"] == season)]
    
    if home_standing.empty or away_standing.empty:
        return 0.0, 0.0
    
    home_winpct = home_standing["WinPCT"].values[0] if "WinPCT" in home_standing.columns else 0.5
    away_winpct = away_standing["WinPCT"].values[0] if "WinPCT" in away_standing.columns else 0.5
    
    home_rank = home_standing["PlayoffRank"].values[0] if "PlayoffRank" in home_standing.columns else 15
    away_rank = away_standing["PlayoffRank"].values[0] if "PlayoffRank" in away_standing.columns else 15
    
    return home_winpct - away_winpct, home_rank - away_rank


def _get_starters_plusminus_diff(home_team_id, away_team_id, game_date, player_df, n=20, top_k=5):
    """
    Calcula el diferencial de +/- promedio de los 'titulares' (top_k jugadores con más minutos)
    de cada equipo en los últimos n partidos antes de game_date.
    Retorna el diferencial o 0 si no hay datos suficientes.
    """
    if player_df is None or player_df.empty:
        return 0.0
    
    game_date = pd.to_datetime(game_date)
    
    # Filtrar partidos previos a la fecha del partido
    past_players = player_df[player_df["GAME_DATE"] < game_date]
    
    # Local: top_k jugadores con más minutos en los últimos n partidos del equipo
    home_players = past_players[past_players["Team_ID"] == home_team_id]
    # Tomar los últimos n game_ids únicos del equipo
    home_game_ids = home_players["Game_ID"].drop_duplicates().tail(n)
    home_recent = home_players[home_players["Game_ID"].isin(home_game_ids)]
    
    if home_recent.empty:
        return 0.0
    
    # Top k por minutos totales
    home_top = home_recent.groupby("PLAYER_ID").agg({"MIN": "sum", "PLUS_MINUS": "mean"}).nlargest(top_k, "MIN")
    home_avg_pm = home_top["PLUS_MINUS"].mean()
    
    # Visitante
    away_players = past_players[past_players["Team_ID"] == away_team_id]
    away_game_ids = away_players["Game_ID"].drop_duplicates().tail(n)
    away_recent = away_players[away_players["Game_ID"].isin(away_game_ids)]
    
    if away_recent.empty:
        return 0.0
    
    away_top = away_recent.groupby("PLAYER_ID").agg({"MIN": "sum", "PLUS_MINUS": "mean"}).nlargest(top_k, "MIN")
    away_avg_pm = away_top["PLUS_MINUS"].mean()
    
    return home_avg_pm - away_avg_pm


def compute_advanced_stats(gameLog_df):
    home_numeric_fields = ["Local_FgPct", "Local_FG3aPct", "Local_FtPct", "Local_BLK", "Local_STL", "Local_AST", "Local_OREB", "Local_DREB", "Local_PF", "Local_PTS","Local_AST_rate", "Local_OREB_rate", "Local_DREB_rate", "Local_STL_rate", "Local_BLK_rate", "Local_PF_rate", "Local_PTS_rate"]
    away_numeric_fields = ["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct", "Visitor_BLK", "Visitor_STL", "Visitor_AST", "Visitor_OREB", "Visitor_DREB", "Visitor_PF", "Visitor_PTS","Visitor_AST_rate", "Visitor_OREB_rate", "Visitor_DREB_rate", "Visitor_STL_rate", "Visitor_BLK_rate", "Visitor_PF_rate", "Visitor_PTS_rate"]
    diff_numeric_fields = ["FGpct_diff", "FG3apct_diff", "FTpct_diff"]
    rate_numeric_fields = ["AST_rate_diff", "OREB_rate_diff", "DREB_rate_diff", "STL_rate_diff", "BLK_rate_diff", "PF_rate_diff", "PTS_rate_diff"]

    # Cálculo de posesiones usando division segura
    local_oreb_pct = _safe_divide(gameLog_df["Local_OREB"], gameLog_df["Local_OREB"] + gameLog_df["Visitor_DREB"])
    gameLog_df["Local_POSS"] = gameLog_df["Local_FGA"] + 0.4 * gameLog_df["Local_FTA"] - 1.07 * local_oreb_pct * (gameLog_df["Local_FGA"] - gameLog_df["Local_FGM"]) + gameLog_df["Local_TOV"]
    
    visitor_oreb_pct = _safe_divide(gameLog_df["Visitor_OREB"], gameLog_df["Visitor_OREB"] + gameLog_df["Local_DREB"])
    gameLog_df["Visitor_POSS"] = gameLog_df["Visitor_FGA"] + 0.4 * gameLog_df["Visitor_FTA"] - 1.07 * visitor_oreb_pct * (gameLog_df["Visitor_FGA"] - gameLog_df["Visitor_FGM"]) + gameLog_df["Visitor_TOV"]

    gameLog_df["Local_FGMissed"] = gameLog_df["Local_FGA"] - gameLog_df["Local_FGM"]
    gameLog_df["Visitor_FGMissed"] = gameLog_df["Visitor_FGA"] - gameLog_df["Visitor_FGM"]

    gameLog_df = rate_calculator(gameLog_df, ["Visitor_AST","Visitor_PTS","Local_STL","Local_PF","Local_BLK"], "Visitor_POSS", [])
    gameLog_df = rate_calculator(gameLog_df, ["Visitor_BLK","Visitor_STL","Visitor_PF","Local_AST","Local_PTS"], "Local_POSS", [])
    gameLog_df = rate_calculator(gameLog_df, ["Local_DREB", "Visitor_OREB"], "Visitor_FGMissed", [])
    gameLog_df = rate_calculator(gameLog_df, ["Visitor_DREB","Local_OREB"], "Local_FGMissed", [])
    
    # Sanitizar cualquier NaN/Inf residual en columnas de rate
    rate_cols = [c for c in gameLog_df.columns if c.endswith("_rate")]
    gameLog_df[rate_cols] = gameLog_df[rate_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    return gameLog_df, home_numeric_fields, away_numeric_fields, diff_numeric_fields, rate_numeric_fields

#funcion principal para obtener los features del historial de partidos
def GameLog_features(n, proceded_path, home_team_id, away_team_id, standings_path=None, player_gamelogs_path=None):
        gameLog_df = pd.read_csv(proceded_path, index_col=False)
        gameLog_df, home_numeric_fields, away_numeric_fields, diff_numeric_fields, rate_numeric_fields = compute_advanced_stats(gameLog_df)
        
        # Cargar datos auxiliares si existen
        standings_df = pd.read_csv(standings_path) if standings_path and os.path.exists(standings_path) else None
        player_df = pd.read_csv(player_gamelogs_path) if player_gamelogs_path and os.path.exists(player_gamelogs_path) else None
        if player_df is not None and "GAME_DATE" in player_df.columns:
            player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"])
        
        # Determinar fecha más reciente para standings
        gameLog_df["GAME_DATE"] = pd.to_datetime(gameLog_df["GAME_DATE"])
        latest_date = gameLog_df["GAME_DATE"].max()

        #selecciono los ultimos n partidos de cada equipo como local/visitante respectivamente
        home_team_df = gameLog_df[gameLog_df["Local_team_id"] == home_team_id].sort_values(by="Game_ID").tail(n)[home_numeric_fields]
        away_team_df = gameLog_df[gameLog_df["Visitor_team_id"] == away_team_id].sort_values(by="Game_ID").tail(n)[away_numeric_fields]
    
        home_team_df = home_team_df.mean().to_frame().T
        away_team_df = away_team_df.mean().to_frame().T

        all_feature_cols = diff_numeric_fields + rate_numeric_fields + ["WinRate_diff", "Standings_WinPct_diff", "Standings_Rank_diff", "Starters_PlusMinus_diff"]
        features = pd.DataFrame(columns=all_feature_cols)
        
        home_diff_stats = home_team_df[["Local_FgPct", "Local_FG3aPct", "Local_FtPct"]].mean().values
        away_diff_stats = away_team_df[["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct"]].mean().values
        features.loc[0, diff_numeric_fields] = home_diff_stats - away_diff_stats
        
        home_rate_stats = home_team_df[["Local_AST_rate", "Local_OREB_rate", "Local_DREB_rate", "Local_STL_rate", "Local_BLK_rate", "Local_PF_rate", "Local_PTS_rate"]].mean().values
        away_rate_stats = away_team_df[["Visitor_AST_rate", "Visitor_OREB_rate", "Visitor_DREB_rate", "Visitor_STL_rate", "Visitor_BLK_rate", "Visitor_PF_rate", "Visitor_PTS_rate"]].mean().values
        features.loc[0, rate_numeric_fields] = home_rate_stats - away_rate_stats
    
        # WinRate diferencial
        home_games_for_wr = gameLog_df[gameLog_df["Local_team_id"] == home_team_id].sort_values(by="Game_ID").tail(n)
        away_games_for_wr = gameLog_df[gameLog_df["Visitor_team_id"] == away_team_id].sort_values(by="Game_ID").tail(n)
        local_winrate = (home_games_for_wr["Local_WL"] == "W").mean() if len(home_games_for_wr) > 0 else 0.5
        visitor_winrate = (away_games_for_wr["Visitor_WL"] == "W").mean() if len(away_games_for_wr) > 0 else 0.5
        features.loc[0, "WinRate_diff"] = local_winrate - visitor_winrate
    
        # Standings features
        winpct_diff, rank_diff = _get_standings_features(home_team_id, away_team_id, latest_date, standings_df)
        features.loc[0, "Standings_WinPct_diff"] = winpct_diff
        features.loc[0, "Standings_Rank_diff"] = rank_diff
        
        # Starters +/- feature
        starters_pm_diff = _get_starters_plusminus_diff(home_team_id, away_team_id, latest_date, player_df, n=n)
        features.loc[0, "Starters_PlusMinus_diff"] = starters_pm_diff
    
        return features

def generate_training_dataset(proceded_path, n, output_path, standings_path=None, player_gamelogs_path=None):
    gameLog_df = pd.read_csv(proceded_path, index_col=False)
    gameLog_df["GAME_DATE"] = pd.to_datetime(gameLog_df["GAME_DATE"])
    gameLog_df = gameLog_df.sort_values(by="GAME_DATE")
    
    gameLog_df, home_numeric_fields, away_numeric_fields, diff_numeric_fields, rate_numeric_fields = compute_advanced_stats(gameLog_df)
    
    # Cargar datos auxiliares si existen
    standings_df = pd.read_csv(standings_path) if standings_path and os.path.exists(standings_path) else None
    player_df = pd.read_csv(player_gamelogs_path) if player_gamelogs_path and os.path.exists(player_gamelogs_path) else None
    if player_df is not None and "GAME_DATE" in player_df.columns:
        player_df["GAME_DATE"] = pd.to_datetime(player_df["GAME_DATE"])
    
    features_list = []
    
    #toda esta parte del for se podria pasar a una nueva funcion de creacion del dataframe de features
    #
    for idx, row in gameLog_df.iterrows():
        game_date = row["GAME_DATE"]
        home_id = row["Local_team_id"]
        away_id = row["Visitor_team_id"]
        
        home_past = gameLog_df[(gameLog_df["Local_team_id"] == home_id) & (gameLog_df["GAME_DATE"] < game_date)].tail(n)
        away_past = gameLog_df[(gameLog_df["Visitor_team_id"] == away_id) & (gameLog_df["GAME_DATE"] < game_date)].tail(n)
        
        if len(home_past) < n or len(away_past) < n:
            continue
            
        home_diff = home_past[["Local_FgPct", "Local_FG3aPct", "Local_FtPct"]].mean().values
        away_diff = away_past[["Visitor_FgPct", "Visitor_FG3aPct", "Visitor_FtPct"]].mean().values
        diff_stats = home_diff - away_diff
        
        home_rates = home_past[["Local_AST_rate", "Local_OREB_rate", "Local_DREB_rate", "Local_STL_rate", "Local_BLK_rate", "Local_PF_rate", "Local_PTS_rate"]].mean().values
        away_rates = away_past[["Visitor_AST_rate", "Visitor_OREB_rate", "Visitor_DREB_rate", "Visitor_STL_rate", "Visitor_BLK_rate", "Visitor_PF_rate", "Visitor_PTS_rate"]].mean().values
        rate_stats = home_rates - away_rates
        
        # WinRate diferencial de los últimos n partidos
        local_winrate = (home_past["Local_WL"] == "W").mean()
        visitor_winrate = (away_past["Visitor_WL"] == "W").mean()
        winrate_diff = local_winrate - visitor_winrate
        
        # Standings features
        winpct_diff, rank_diff = _get_standings_features(home_id, away_id, game_date, standings_df)
        
        # Starters +/- feature
        starters_pm_diff = _get_starters_plusminus_diff(home_id, away_id, game_date, player_df, n=n)
        
        target = 1 if row["Local_WL"] == "W" else 0
        
        feature_row = {
            "Game_ID": row["Game_ID"],
            "GAME_DATE": row["GAME_DATE"],
            "Local_team_id": home_id,
            "Visitor_team_id": away_id,
            "Target_Local_Win": target
        }
        
        for i, col in enumerate(diff_numeric_fields):
            feature_row[col] = diff_stats[i]
        for i, col in enumerate(rate_numeric_fields):
            feature_row[col] = rate_stats[i]
        feature_row["WinRate_diff"] = winrate_diff
        feature_row["Standings_WinPct_diff"] = winpct_diff
        feature_row["Standings_Rank_diff"] = rank_diff
        feature_row["Starters_PlusMinus_diff"] = starters_pm_diff
            
        features_list.append(feature_row)
        
    training_df = pd.DataFrame(features_list)
    
    # Sanitizar NaN/Inf en el dataset final
    numeric_cols = training_df.select_dtypes(include=[np.number]).columns
    training_df[numeric_cols] = training_df[numeric_cols].replace([np.inf, -np.inf], 0).fillna(0)
    
    training_df.to_csv(output_path, index=False)
    return training_df




def rate_calculator(gameLog_df, stats_cols : list, reference : str, lista) -> pd.DataFrame:

    """
        calcula el atributo rate para cualquier estadistica pasada en stats_cols en relacion al parametro reference

        reference puede ser "Local_POSS" o "Visitor_POSS" dependiendo si se quiere calcular el rate para el equipo local o visitante
        tambien puede ser "Visitor_FGMissed" o "Local_FGMissed" si se quiere calcular el rate en relacion a los tiros fallados del equipo contrario

        devuelve el data frame con las nuevas columnas calculadas
    """
    df = gameLog_df.copy()

    for stat in stats_cols:
        df[stat + "_rate"] = _safe_divide(df[stat], df[reference])


    return df
