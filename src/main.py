from data_eng import data_collection as dc
from data_eng import data_cleaning as dCl
from data_an import feature_eng as eng
from data_sci import prob_model as model
import pandas as pd
import os
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def menu():
    clear_screen()
    print("\n" + "="*70)
    print(r"""
    _   _ ____    _       __  __ _       
   | \ | |  _ \  / \     |  \/  | |      
   |  \| | |_) |/ _ \    | |\/| | |      
   | |\  |  _ </ ___ \   | |  | | |___   
   |_| \_|_| \/_/   \_\  |_|  |_|_____|  
                                         
          PREDICTOR ESTADÍSTICO 🏀      
    """)
    print("="*70)
    print("  [1] Predecir un partido de hoy")
    print("  [2] Entrenar modelo (Auto-Updater en Segundo Plano)")
    print("  [3] Actualizar Base de Datos Maestra (Descarga manual)")
    print("  [4] Salir del Sistema")
    print("="*70)
    
    opcion = input("\n» Seleccione una opción (1-4): ")
    return opcion

def predict_todays_game():
    print("\nObteniendo partidos del día...")
    # 1. Obtener partidos del día usando scoreboardv2
    game_days_df = dc.fetch_gameDays()
    
    if game_days_df is None or game_days_df.empty:
        print("No se encontraron partidos para el día de hoy o hubo un error en la API.")
        return
        
    # Filtrar columnas relevantes del scoreboard (nombres o IDs de equipo)
    # Según la NBA API, GAME_DATE_EST, y en general HOME_TEAM_ID y VISITOR_TEAM_ID están disponibles
    # pero revisemos las columnas reales que devuelve
    available_cols = game_days_df.columns.tolist()
    
    # Trataremos de identificar HOME_TEAM_ID y VISITOR_TEAM_ID
    if "HOME_TEAM_ID" in available_cols and "VISITOR_TEAM_ID" in available_cols:
        games = game_days_df[["GAME_ID", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]].dropna().drop_duplicates()
        
        print("\n[+] Partidos disponibles hoy:")
        games_list = []
        for idx, row in games.iterrows():
            # Necesitamos nombres basándonos en ID
            from nba_api.stats.static import teams
            home_team = teams.find_team_name_by_id(row["HOME_TEAM_ID"])
            away_team = teams.find_team_name_by_id(row["VISITOR_TEAM_ID"])
            
            home_name = home_team["full_name"] if home_team else str(row["HOME_TEAM_ID"])
            away_name = away_team["full_name"] if away_team else str(row["VISITOR_TEAM_ID"])
            
            games_list.append((row["HOME_TEAM_ID"], row["VISITOR_TEAM_ID"], home_name, away_name))
            print(f"  {len(games_list)}. {away_name} @ {home_name}")
            
        if not games_list:
            print("[!] No se pudieron procesar correctamente los partidos del día.")
            return
            
        try:
            seleccion = int(input(f"\n» Elija un número de partido (1-{len(games_list)}): "))
            if 1 <= seleccion <= len(games_list):
                home_id, away_id, home_name, away_name = games_list[seleccion - 1]
                
                print(f"\n[*] Generando features históricas para: {away_name} @ {home_name}...")
                featured_gameLog_df = eng.GameLog_features(
                    20,
                    os.path.join(project_root,"data","procesed_data","gamelogs_clean.csv"),
                    home_id, 
                    away_id
                )
                
                print("[*] Cargando modelo de Regresión Logística...")
                model_path = os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl")
                if not os.path.exists(model_path):
                    print(f"[!] No se encontró un modelo entrenado. Por favor, realiza un Entrenamiento Continuo primero.")
                    return
                    
                logistic_model = model.load_model(model_path)
                
                print("[*] Realizando predicción...")
                prob_victoria = model.predict_game(logistic_model, featured_gameLog_df)
                
                fecha_partido = datetime.now().strftime("%Y-%m-%d")
                
                pred_text = "\n" + "-"*60 + "\n"
                pred_text += f"   📊 PREDICCIÓN DE PARTIDO (Fecha: {fecha_partido})\n"
                pred_text += "-"*60 + "\n"
                pred_text += f"   🏠 Local:     {home_name}\n"
                pred_text += f"   ✈️ Visitante: {away_name}\n\n"
                pred_text += f"   🏆 Prob. Victoria Local ({home_name}): {prob_victoria*100:.2f}%\n"
                pred_text += f"   🛑 Prob. Victoria Visit ({away_name}): {(1-prob_victoria)*100:.2f}%\n"
                pred_text += "-"*60 + "\n"
                
                print(pred_text)
                
                reports_dir = os.path.join(project_root, "reports")
                os.makedirs(reports_dir, exist_ok=True)
                pred_report_path = os.path.join(reports_dir, "latest_prediction_report.txt")
                
                with open(pred_report_path, "a", encoding="utf-8") as f:
                    f.write(pred_text)
                print(f"Informe de predicción añadido (append) en: {pred_report_path}")
                
                input("\nPresione ENTER para volver al menú principal...")
                
            else:
                print("Selección inválida.")
                input("\nPresione ENTER para volver al menú principal...")
        except ValueError:
            print("Por favor, ingrese un número válido.")
            input("\nPresione ENTER para volver al menú principal...")
    else:
        print("El retorno de la API del scoreboard no contiene las columnas necesarias (HOME_TEAM_ID, VISITOR_TEAM_ID).")
        print("Columnas disponibles:", available_cols)
        input("\nPresione ENTER para volver al menú principal...")

def update_master_data():
    clear_screen()
    print("\n" + "="*60)
    print(" 📡 INICIANDO RECARGA DE BASE DE DATOS MAESTRA")
    print("="*60)
    print("[*] Este proceso está consultando la API de la NBA.")
    print("[*] Puede tardar varios minutos dependiendo del volumen de datos...")
    try:
        seasons = ["2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
        # dc.fetch_teams_gameLogs guarda en all_teams_gamelogs.csv
        dc.fetch_teams_gameLogs(seasons)
        print("\n[+] Datos crudos descargados con éxito.")
        print("[*] Limpiando y procesando gamelogs para su almacenamiento seguro...")
        
        dCl.gameLog_csv_filter(
            os.path.join(project_root,"data","raw_data","all_teams_gamelogs.csv"),
            os.path.join(project_root,"data","procesed_data","gamelogs_clean.csv")
        )
        print("\n[✔] ¡Base de Datos Maestra actualizada exitosamente!")
        input("\nPresiona ENTER para volver al menú...")
    except Exception as e:
        print(f"\n[X] Error crítico al actualizar la base de datos maestra: {e}")
        input("\nPresiona ENTER para volver al menú...")

def continuous_training_loop():
    clear_screen()
    print("\n" + "="*60)
    print(" 🧠 INICIALIZANDO ENTRENAMIENTO CONTINUO ")
    print(" (INFO: Presiona Ctrl + C en cualquier momento para detener)")
    print("="*60)
    
    training_dataset_path = os.path.join(project_root, "data", "procesed_data", "training_dataset.csv")
    cleaned_gamelogs_path = os.path.join(project_root, "data", "procesed_data", "gamelogs_clean.csv")
    
    # 1. Cargar datos históricos base para obtener la fecha más reciente
    if not os.path.exists(cleaned_gamelogs_path):
        print("Error: No existe gamelogs_clean.csv. Por favor, corre la Opción 3 primero.")
        return
        
    master_df = pd.read_csv(cleaned_gamelogs_path)
    latest_date_str = master_df["GAME_DATE"].max()
    
    print(f"[*] Última fecha en la base de datos: {latest_date_str}")
    
    # Métricas iniciales
    try:
        X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
        logistic_model = model.train_logistic_model(X_train, y_train)
        acc_base, _ = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
    except Exception as e:
        print("[X] No se pudo evaluar el modelo inicial. ¿Existe el training dataset?")
        return
        
    partidos_analizados = 0
    
    import sys
    import time
    
    try:
        while True:
            # UI Dinámica (Al iniciar asume que contactará rápido)
            sys.stdout.write(f"\rBuscando desde {latest_date_str}... Asimilados: {partidos_analizados} | Status: Conectando a API NBA...          ")
            sys.stdout.flush()
            
            new_games = dc.fetch_incremental_gamelogs(latest_date=latest_date_str)
            
            if new_games is not None and not new_games.empty:
                sys.stdout.write(f"\rBuscando partidos nuevos desde {latest_date_str}... Partidos asimilados: {partidos_analizados} | Status: Procesando {len(new_games)} files...")
                sys.stdout.flush()
                
                # 1. Necesitamos filtrar y limpiar esos nuevos raw gamelogs e introducirlos al dataset limpio maestro
                dCl.gameLog_csv_filter(
                    os.path.join(project_root,"data","raw_data","all_teams_gamelogs.csv"),
                    os.path.join(project_root,"data","procesed_data","gamelogs_clean.csv")
                )
                
                # 2. Re-generar un pequeño training dataset SÓLO con esos partidos nuevos
                # (Para no cambiar la firma original completa de feature_eng ahora, 
                # usaremos append llamando a un script interno modificado o rehaciendo el histórico)
                sys.stdout.write(f"\rBuscando partidos nuevos desde {latest_date_str}... Partidos asimilados: {partidos_analizados} | Status: Actualizando Features...")
                sys.stdout.flush()
                eng.generate_training_dataset(
                    cleaned_gamelogs_path, 20, training_dataset_path
                )
                
                # 3. Re-entrenar modelo matemático
                sys.stdout.write(f"\rBuscando partidos nuevos desde {latest_date_str}... Partidos asimilados: {partidos_analizados} | Status: Re-Entrenando...")
                sys.stdout.flush()
                
                X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
                logistic_model = model.train_logistic_model(X_train, y_train)
                model_path = os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl")
                model.save_model(logistic_model, model_path, verbose=False)
                
                # Actualizar contadores
                partidos_analizados += len(new_games) // 2 # Porque cada partido tiene 2 gamelogs
                
                # Recargar la latest date leída del nuevo CSV
                updated_master_df = pd.read_csv(cleaned_gamelogs_path)
                latest_date_str = updated_master_df["GAME_DATE"].max()
                
            else:
                 sys.stdout.write(f"\rBuscando desde {latest_date_str}... Asimilados: {partidos_analizados} | Status: Sin juegos nuevos. Durmiendo 1 hr...")
                 sys.stdout.flush()
                 time.sleep(3600)

    except KeyboardInterrupt:
        print("\n\n" + "="*50)
        print(" REPORTE DE ENTRENAMIENTO FINALIZADO ")
        print("="*50)
        print(f"- Nuevos partidos procesados: {partidos_analizados}")
        
        # Cargar último estado e imprimir delta de métricas
        X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
        print(f"   - Tamaño actual del dataset: {len(X_train) + len(X_test)} partidos históricos")
        print(f"   - Precisión base inicial:    {acc_base * 100:.2f}%")
        
        acc_final, brier_final = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
        
        delta = acc_final - acc_base
        signo = "+" if delta >= 0 else ""
        print(f"   - Nueva precisión lograda:   {acc_final * 100:.2f}%")
        print(f"   - Mejora Total de Precisión: {signo}{delta * 100:.2f}%\n")
        print(f"[*] El conocimiento algorítmico se ha actualizado automáticamente.")
        input("\nPresione ENTER para salir de vuelta al menú...")
        return

def main():
    while True:
        opc = menu()
        if opc == '1':
            predict_todays_game()
        elif opc == '2':
            continuous_training_loop()
        elif opc == '3':
            update_master_data()
        elif opc == '4':
            print("\nSaliendo del programa. ¡Hasta luego!")
            break
        else:
            print("\nOpción no válida. Intente de nuevo.")

if __name__ == "__main__":
    main()