import pandas as pd
import numpy as np
import os
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, brier_score_loss, confusion_matrix, classification_report

def load_and_split_data(dataset_path, season_start_year=2015):
    """
    Carga el dataset de entrenamiento, filtra por temporadas desde season_start_year
    hasta la temporada anterior a la actual, separa features (X) de target (y)
    y divide en conjuntos de entrenamiento y prueba.
    """
    df = pd.read_csv(dataset_path)
    
    # --- Filtro temporal: desde temporada season_start_year hasta la temporada anterior ---
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    
    # Inicio: octubre del año season_start_year (inicio temporada NBA)
    fecha_inicio = pd.Timestamp(f"{season_start_year}-10-01")
    
    # Fin: calcular la temporada anterior a la actual dinámicamente
    now = datetime.now()
    if now.month >= 10:
        # Estamos en la temporada que empezó este año → la anterior terminó en junio de este año
        fecha_fin = pd.Timestamp(f"{now.year}-06-30")
    else:
        # Estamos en la temporada que empezó el año pasado → la anterior terminó en junio del año pasado
        fecha_fin = pd.Timestamp(f"{now.year - 1}-06-30")
    
    df = df[(df["GAME_DATE"] >= fecha_inicio) & (df["GAME_DATE"] <= fecha_fin)]
    
    if df.empty:
        raise ValueError(f"No hay datos en el rango {fecha_inicio.date()} - {fecha_fin.date()}. "
                         f"Verifica que el training_dataset contenga partidos en esas fechas.")
    
    # Separamos y (Target)
    y = df['Target_Local_Win'].values
    
    # Separamos X (Features). Descartamos columnas informativas categóricas.
    cols_to_drop = ['Game_ID', 'GAME_DATE', 'Local_team_id', 'Visitor_team_id', 'Target_Local_Win']
    X = df.drop(columns=cols_to_drop)
    
    # Dividimos 80% entrenamiento, 20% prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, X.columns

def train_logistic_model(X_train, y_train, use_gridsearch=True, verbose=True):
    """
    Crea un pipeline que estandariza los datos y entrena una regresión logística.
    Si use_gridsearch=True, realiza GridSearchCV para encontrar los mejores hiperparámetros.
    Retorna el mejor pipeline (modelo) y los mejores parámetros encontrados.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000))
    ])
    
    if use_gridsearch:
        param_grid = {
            'classifier__C': [0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['saga', 'lbfgs'],
        }
        
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            error_score='raise'
        )
        
        try:
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            if verbose:
                print(f"\n   [GridSearch] Mejor accuracy CV: {best_score*100:.2f}%")
                print(f"   [GridSearch] Mejores parámetros: {best_params}")
            
            return best_model, best_params
            
        except Exception as e:
            if verbose:
                print(f"\n   [GridSearch] Error en la búsqueda: {e}")
                print(f"   [GridSearch] Usando parámetros por defecto...")
            pipeline.fit(X_train, y_train)
            return pipeline, {"fallback": True, "error": str(e)}
    else:
        pipeline.fit(X_train, y_train)
        return pipeline, {"gridsearch": False}

def evaluate_model(model, X_test, y_test, report_path=None, verbose=True):
    """
    Evalúa el modelo usando el conjunto de prueba e imprime métricas clave.
    Si se especifica un report_path, guardará un archivo de texto con la documentación.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    report_text = "--- Resultados de la Evaluación del Modelo ---\n"
    report_text += f"Accuracy (Precisión Global): {acc:.4f}\n"
    report_text += f"Brier Score (Puntuación de Calibración): {brier:.4f}\n\n"
    report_text += "Matriz de Confusión:\n"
    report_text += str(confusion_matrix(y_test, y_pred)) + "\n\n"
    report_text += "Reporte de Clasificación:\n"
    report_text += classification_report(y_test, y_pred) + "\n"
    
    if verbose:
        print(report_text)
    
    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        if verbose:
            print(f"[+] Informe de evaluación guardado en: {report_path}")
    
    return acc, brier

def save_model(model, filepath, verbose=True):
    """
    Guarda el modelo entrenado en disco.
    """
    joblib.dump(model, filepath)
    if verbose:
        print(f"[+] Modelo guardado exitosamente en: {filepath}")

def load_model(filepath):
    """
    Carga un modelo desde el disco.
    """
    return joblib.load(filepath)

def predict_game(model, match_features_df):
    """
    Dada una fila de estadísticas generada, calcula la probabilidad de victoria local.
    """
    prob_win = model.predict_proba(match_features_df)[0][1]
    return prob_win