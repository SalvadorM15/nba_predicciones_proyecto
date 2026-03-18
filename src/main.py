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

            #SELECCION Y CREACION DE FEATURES PARA EL PARTIDO A PREDECIR
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
                    
                #CARGA DEL MODELO Y PREDICCION ESTADISTICA
                logistic_model = model.load_model(model_path)
                
                print("[*] Realizando predicción...")
                prob_victoria = model.predict_game(logistic_model, featured_gameLog_df)
                
                fecha_partido = datetime.now().strftime("%Y-%m-%d")


                #IMPRESION DE RESULTADOS EN LA PREDICCION
                pred_text = "\n" + "-"*60 + "\n"
                pred_text += f"   📊 PREDICCIÓN DE PARTIDO (Fecha: {fecha_partido})\n"
                pred_text += "-"*60 + "\n"
                pred_text += f"   🏠 Local:     {home_name}\n"
                pred_text += f"   ✈️ Visitante: {away_name}\n\n"
                pred_text += f"   🏆 Prob. Victoria Local ({home_name}): {prob_victoria*100:.2f}%\n"
                pred_text += f"   🛑 Prob. Victoria Visit ({away_name}): {(1-prob_victoria)*100:.2f}%\n"
                pred_text += "-"*60 + "\n"
                
                print(pred_text)
                
                #PERSISTENCIA DE LOS DATOS DE PREDICCION
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






#NO MUCHO MAS QUE APORTAR ACA, MODELO YA ENTRENADO CON PIPELINE Y GRID_SEARCH
def continuous_training_loop():
    clear_screen()
    print("\n" + "="*60)
    print(" 🧠 INICIALIZANDO ENTRENAMIENTO COMPLETO (2021 - Presente) ")
    print(" (INFO: Este proceso entrenará el modelo con todos los datos disponibles)")
    print("presionar ctrl + c para terminar el entrenamiento en cualquier momento")
    print("="*60)
    
    training_dataset_path = os.path.join(project_root, "data", "procesed_data", "training_dataset.csv")
    cleaned_gamelogs_path = os.path.join(project_root, "data", "procesed_data", "gamelogs_clean.csv")

    # 1. Verificar datos históricos base 
    if not os.path.exists(cleaned_gamelogs_path):
        print("Error: No existe gamelogs_clean.csv. Por favor, corre la Opción 3 primero.")
        input("\nPresione ENTER para volver al menú...")
        return
        
    master_df = pd.read_csv(cleaned_gamelogs_path)
    #latest_date_str = master_df["GAME_DATE"].max()

    #print(f"[*] Última fecha en la base de datos maestra: {latest_date_str}")
    print("[*] Generando/Actualizando dataset de entrenamiento con TODOS los partidos (esto puede tardar unos minutos)...")
    
    # 2. Variable para almacenar el *accuracy* y el modelo
    model_path = os.path.join(project_root, "data", "procesed_data", "nba_logistic_model.pkl")
    initial_acc = 0.0
    logistic_model = None

    print("\n[*] Preparando entorno de Machine Learning...")
    start_time = time.time()
    
    try:
        ciclos = 0
        while True:
            ciclos += 1
            print(f"\n--- [CICLO de ENTRENAMIENTO #{ciclos}] ---")

            try:
                # Opcional (pero clave para "crecimiento"): Re-generar / re-cargar todo el dataset por si hay partidos nuevos en segundo plano.
                # Como esto puede tardar un poco, asumimos que X_train y y_train se actualizan si es necesario.
                # Para simplificar, recargamos el split actual cada ciclo:
                X_train, X_test, y_train, y_test, _ = model.load_and_split_data(training_dataset_path, season_start_year=2021)
                
                if ciclos == 1:
                    print(f"[*] Datos cargados exitosamente (Tamaño total: {len(X_train) + len(X_test)} partidos).")
                
                # Cargamos el último modelo guardado si existe, tal como solicitaste para continuar el entrenamiento.
                if os.path.exists(model_path):
                    logistic_model = model.load_model(model_path)
                    
                    if ciclos == 1:
                        initial_acc, _ = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
                        print(f"[*] Precisión histórica previa: {initial_acc*100:.2f}%")

                if logistic_model is not None and ciclos > 1:
                    sys.stdout.write(f"[*] Re-utilizando el modelo guardado del ciclo {ciclos-1}...\n")
                    logistic_model, params = model.train_logistic_model(X_train, y_train, use_gridsearch=False, verbose=False, existing_pipeline=logistic_model)
                else:
                    sys.stdout.write("[*] Buscando parámetros óptimos desde cero y validando hiperparámetros...\n")
                    logistic_model, params = model.train_logistic_model(X_train, y_train, use_gridsearch=True, verbose=True)
                
                print(f"[*] Modelo entrenado exitosamente.")
                acc_final, brier_final = model.evaluate_model(logistic_model, X_test, y_test, verbose=False)
                
                # Mostrar mejora inmediata respecto a la iteración anterior
                delta = acc_final - initial_acc
                signo = "+" if delta >= 0 else ""
                print(f"[*] Precisión alcanzada: {acc_final*100:.2f}% ({signo}{delta*100:.2f}%)")
                
                # Guardamos como el último y mejor modelo para la próxima vuelta
                model.save_model(logistic_model, model_path, verbose=False)
                
                # También persistir hiperparámetros
                best_params_path = os.path.join(project_root, "data", "procesed_data", "best_params.json")
                import json
                with open(best_params_path, "w") as f:
                    json.dump(params, f)
                    
                # El acc_final de este ciclo se convierte en el initial_acc del próximo
                initial_acc = acc_final
                
                print("[*] Ciclo completado. Esperando 5 segundos antes de leer los datos de nuevo...")
                time.sleep(5)
                    
            except Exception as e:
                print("\n[X] Error crítico durante el entrenamiento y la evaluación del modelo.")
                print(f"Traza: {e}")
                input("\nPresione ENTER para salir de vuelta al menú...")
                return
    except KeyboardInterrupt:    
        end_time = time.time()
        tt = end_time - start_time
        
        print("\n" + "="*50)
        print(" ENTRENAMIENTO COMPLETADO ")
        print("="*50)
        print(f"   - Minutos invertidos:        {tt/60:.2f} min")
        print(f"   - Tamaño actual del dataset: {len(X_train) + len(X_test)} partidos históricos")
        print(f"   - Última Precisión:          {acc_final * 100:.2f}%")
        print(f"   - Brier Score (Calibración): {brier_final:.4f}\n")
        print(f"[*] El conocimiento algorítmico se ha actualizado automáticamente al máximo absoluto.")
            
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