from data_eng import data_collection as dc
from data_eng import data_cleaning as dCl
from data_an import feature_eng as eng
from data_sci import prob_model as model
import pandas as pd
import os
from datetime import datetime
import sys
import time
import pickle

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
        seasons = ["2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25"]
        # dc.fetch_teams_gameLogs guarda en all_teams_gamelogs.csv
        dc.fetch_teams_gameLogs(seasons)
        print("\n[+] Datos crudos descargados con éxito.")
        print("[*] Limpiando y procesando gamelogs para su almacenamiento seguro...")
        
        dCl.gameLog_csv_filter(
            os.path.join(project_root,"data","raw_data","all_teams_gamelogs.csv"),
            os.path.join(project_root,"data","procesed_data","gamelogs_clean.csv")
        )

        print("datos actualizados: ",pd.read_csv(os.path.join(project_root,"data","procesed_data","gamelogs_clean.csv")).tail())
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
    master_df = master_df[master_df["GAME_DATE"] < "2022-01-01"]
    latest_date_str = master_df["GAME_DATE"].max()

    eng.generate_training_dataset(cleaned_gamelogs_path, 20, training_dataset_path)
            
    # Leer el set actualizado (que ahora incluye la fila del partido iterado)
    new_train_df = pd.read_csv(training_dataset_path)
    print(new_train_df.head())
    print(f"[*] Última fecha en la base de datos: {latest_date_str}")
    
    # Métricas iniciales
    try:
        #print("[*] Cargando datos históricos...")
        X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
        print("[*] Datos cargados exitosamente.")
        logistic_model = model.train_logistic_model(X_train, y_train, None)
        print("[*] Modelo entrenado exitosamente.")
        acc_base, _ = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
        model.save_model(logistic_model, os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl"))
        with open(os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl"), "rb") as f:
            model = pickle.load(f)
    
        params = model.get_params();
        
    except Exception as e:
        time.sleep(5)
        print("[X] No se pudo evaluar el modelo inicial. ¿Existe el training dataset?")
        return
        
    partidos_analizados = 0
    
    try:
        # 1. Determinar cuál es el partido más reciente ya asimilado en el training set
        if not os.path.exists(training_dataset_path):
            print("[!] No existe training_dataset.csv. Por favor, asegúrese de generarlo desde un punto de inicio.")
            return
            
        training_df = pd.read_csv(training_dataset_path)
        training_df["GAME_DATE"] = pd.to_datetime(training_df["GAME_DATE"])
        tranining_df = training_df[training_df["GAME_DATE"] < "2022-01-01"]
        
        #print(f"[*] Última fecha entrenada en el dataset: {latest_trained_date.strftime('%Y-%m-%d')}")
        
        # 2. Identificar qué partidos están en master_df pero NO en training_df (o sea, son posteriores)
        master_df["GAME_DATE"] = pd.to_datetime(master_df["GAME_DATE"])
        print("master_df", master_df.head().tail())
        print(master_df[master_df["GAME_DATE"] > latest_trained_date].tail())

        #untrained_games = master_df[master_df["GAME_DATE"] > latest_trained_date].sort_values(by="GAME_DATE", ascending=True)
               
        #print(f"[*] Se encontraron {len(untrained_games)} partidos sin procesar en el histórico.")
        
        # Agrupamos por Game_ID para iterar evento por evento (cada Game_ID tiene 2 filas en gamelogs_clean)
        unique_games = untrained_games.drop_duplicates(subset=["Game_ID"]).sort_values(by="GAME_DATE", ascending=True)
        total_a_procesar = len(unique_games)
        
        for idx, row in unique_games.iterrows():
            game_id = row["Game_ID"]
            game_date_str = row["GAME_DATE"].strftime('%Y-%m-%d')
            home_id = row["Local_team_id"]
            away_id = row["Visitor_team_id"]
            
            sys.stdout.write(f"\rProgreso: {partidos_analizados}/{total_a_procesar} | Fecha actual: {game_date_str} | Status: Generando Features...")
            sys.stdout.flush()
            
            # Necesitamos generar las features JUSTO antes de este partido usando el histórico hasta ese día
            # Podríamos re-llamar al script original, pero es ineficiente leer el master_df completo por cada iteración.
            # En su lugar, usaremos generate_training_dataset para que procese secuencialmente los delta pendientes.
            # Para esto, simularemos un batch pequeño para no rehacer todo:
            
            # Extraer histórico estrictamente anterior a este partido
            historico_hasta_hoy = master_df[master_df["GAME_DATE"] <= row["GAME_DATE"]]
            
            # Guardamos un CSV temporal para que `generate_training_dataset` lo tome como input
            temp_path = os.path.join(project_root, "data", "procesed_data", "temp_gamelogs.csv")
            historico_hasta_hoy.to_csv(temp_path, index=False)
            
            # Generamos features sólo de ese batch temporal (el script interno procesa todo hasta temp_path)
            # Como optimización, generate_training_dataset ya procesa ordenado. Podríamos pasarle directamente
            # el máster y que sobreescriba, pero el usuario pidió 1 por 1 para ver el progreso.
            # Aquí lo ideal sería que generate_training_dataset anexe o procese delta, pero por simplicidad
            # usaremos el enfoque de reconstruir el training hasta el target y entrenar:
            
            # Una mejor aproximación 1 por 1: generamos las features de ese UNICO partido local/visitante
            # (Reutilizamos lógica de eng.compute_advanced_stats internamente pero simplificada llamando al generador completo
            # con el histórico incrementado)
            
            # Nota: para no hacerlo computacionalmente inviable iterar miles en disco, vamos a rehacer las features 
            # del batch faltante y las concatenamos al training base. 
            pass

        # Para cumplir EXACTAMENTE lo pedido "una vez terminada pasa al siguiente partido..."
        # Vamos a reescribir la lógica de update línea por línea:
        
        for idx, row in unique_games.iterrows():
            sys.stdout.write(f"\rBuscando desde {latest_trained_date.strftime('%Y-%m-%d')}... Asimilados: {partidos_analizados}/{total_a_procesar} | Status: Extrayendo Features...")
            sys.stdout.flush()
            
            # Extraer histórico <= a este partido temporalmente
            temp_master = master_df[master_df["GAME_DATE"] <= row["GAME_DATE"]].copy()
            temp_csv = os.path.join(project_root, "data", "procesed_data", "temp_gamelogs.csv")
            temp_master.to_csv(temp_csv, index=False)
            
            # Generar features SOLO guardando temporal (es costoso pero cumple el requerimiento secuencial)
            temp_train_path = os.path.join(project_root, "data", "procesed_data", "temp_train.csv")
            eng.generate_training_dataset(temp_csv, 20, temp_train_path)
            
            # Leer el set actualizado (que ahora incluye la fila del partido iterado)
            new_train_df = pd.read_csv(temp_train_path)
            
            # Lo sobreescribimos al oficial para persistir el avance por partido
            new_train_df.to_csv(training_dataset_path, index=False)
            
            sys.stdout.write(f"\rBuscando desde {latest_trained_date.strftime('%Y-%m-%d')}... Asimilados: {partidos_analizados}/{total_a_procesar} | Status: Re-Entrenando...     ")
            sys.stdout.flush()


            
            # Re-entrenar
            X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
            logistic_model = modexl.train_logistic_model(X_train, y_train, params)
            model_path = os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl")
            model.save_model(logistic_model, model_path, verbose=False)
            
            partidos_analizados += 1


    except KeyboardInterrupt:

        #una vez cortado el entrenamiento elijo el mejor modelo y los mejores parametros y los persisto para reutilizarlos
        best_model, best_params = model.grid_tune_regretion(X_train, y_train)
        model.save_model(best_model, "nba_logistic_model.pkl")
        best_params_path = os.path.join(project_root, "data", "procesed_data", "best_params.json")
        with open(best_params_path, "w") as f:
            json.dump(best_params, f)

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
        print(f"\n[*] El conocimiento algorítmico se ha actualizado automáticamente.")
        
        # Limpiar temporales si existen
        temp_csv = os.path.join(project_root, "data", "procesed_data", "temp_gamelogs.csv")
        temp_train = os.path.join(project_root, "data", "procesed_data", "temp_train.csv")
        if os.path.exists(temp_csv): os.remove(temp_csv)
        if os.path.exists(temp_train): os.remove(temp_train)
        
        input("\nPresione ENTER para salir de vuelta al menú...")
        return
    
    # Si termina el loop normal (sin Ctrl+C)
    print("\n\n" + "="*50)
    print(" ENTRENAMIENTO COMPLETADO ")
    print("="*50)
    print(f"- Todos los {partidos_analizados} partidos históricos pendientes fueron asimilados.")
    
    X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path)
    acc_final, brier_final = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
    delta = acc_final - acc_base
    signo = "+" if delta >= 0 else ""
    
    print(f"   - Tamaño actual del dataset: {len(X_train) + len(X_test)} partidos")
    print(f"   - Mejora Total de Precisión: {signo}{delta * 100:.2f}%\n")
    
    # Limpiar temporales
    temp_csv = os.path.join(project_root, "data", "procesed_data", "temp_gamelogs.csv")
    temp_train = os.path.join(project_root, "data", "procesed_data", "temp_train.csv")
    if os.path.exists(temp_csv): os.remove(temp_csv)
    if os.path.exists(temp_train): os.remove(temp_train)
        
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